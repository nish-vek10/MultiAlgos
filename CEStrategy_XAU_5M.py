"""
Chandelier Exit Strategy for XAUUSD
Using Heikin-Ashi Candles
XAU - 5 Minute TF
"""

# ===== FILENAME: CEStrategy_XAU_5M =====

import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
import os
import requests
import sys
import math
import pandas as pd
from pytz import timezone

# === CONFIGURATION === #
mt5_symbol = "XAUUSD"
oanda_symbol = "XAU_USD"
timeframe = mt5.TIMEFRAME_M5
num_candles = 500
lot_size = 0.1
slippage = 10
use_heikin_ashi = True
atr_period = 1
atr_mult = 1.85
magic_number = 112233  # Unique ID for this EA's trades
local_tz = timezone('Europe/London')

# === RISK & STOPS CONFIG === #
risk_per_trade_pct = 0.25      # percent of balance risked per trade
sl_usd_distance    = 4.0       # fixed $ distance from entry


# === SESSION FILTER (Europe/London local time) === #
session_windows = [
    ("07:00", "12:00"),  # London morning
    ("13:00", "18:00"),  # US session + London/NY overlap
]

# session behaviour
auto_close_at_session_end = True       # close any open position when session ends
require_post_start_candle = True       # only allow first trade after the first candle formed AFTER session start

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52474875
mt5_password = "W7J&K6Zrsimovi"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\52474875\terminal64.exe"

print("MT5 Path Exists?", os.path.exists(mt5_terminal_path))

# === CONNECT TO MT5 === #
if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server, path=mt5_terminal_path):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    quit()

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 ACCOUNT CONNECTED! \nACCOUNT: {account.login} \nBALANCE: ${account.balance:.2f}\n")

symbol_info = mt5.symbol_info(mt5_symbol)
if symbol_info:
    print(f"[INFO] {mt5_symbol} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
    print(symbol_info._asdict())
else:
    print(f"[ERROR] Unable to fetch symbol info for {mt5_symbol}")

oanda_token = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_account_id = "101-004-35770497-001"
oanda_api_url = "https://api-fxpractice.oanda.com/v3"

def fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=num_candles):
    """
    Fetch candle data from OANDA REST API.
    symbol: OANDA instrument name (e.g., "XAU_USD")
    granularity: M1, M5, M15, H1, D etc.
    count: number of candles to fetch
    """
    url = f"{oanda_api_url}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M"  # Midpoint prices
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print("[ERROR] Failed to fetch candles from OANDA:", response.status_code, response.text)
        return None

    raw_candles = response.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}

    for candle in raw_candles:
        if candle.get('complete', False):
            utc_time = pd.to_datetime(candle['time'], utc=True)
            local_time = utc_time.tz_convert(local_tz)
            data['time'].append(local_time)
            data['open'].append(float(candle['mid']['o']))
            data['high'].append(float(candle['mid']['h']))
            data['low'].append(float(candle['mid']['l']))
            data['close'].append(float(candle['mid']['c']))
            data['volume'].append(int(candle['volume']))

    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index("time", inplace=True)
    return df

# === UTILITY FUNCTIONS === #
def get_position(symbol):
    """
    Check for existing position on a symbol (by magic number).
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None

    for p in positions:
        if p.magic == magic_number:
            return p
    return None

def _round_to_step(value, step):
    steps = math.floor(value / step)
    return max(steps * step, 0.0)

def compute_lot_for_risk(symbol, sl_usd, risk_pct):
    """
    Compute lot size so that loss at SL ~= risk_pct% of current balance.
    Uses tick_value/tick_size so it works across brokers.
    Assumes tick_value is for 1.0 lot (standard in MT5).
    """
    acct = mt5.account_info()
    if acct is None:
        raise RuntimeError(f"Account not available: {mt5.last_error()}")
    balance = float(acct.balance)
    risk_amount = balance * (float(risk_pct) / 100.0)
    if risk_amount <= 0:
        return None, 0.0

    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return None, 0.0

    # Value of a 1.0 price unit move for 1.0 lot
    # (e.g., for XAUUSD this is usually 1.0 / 0.01 * tick_value = 100 * tick_value)
    if si.tick_size == 0:
        print("[ERROR] tick_size is zero; cannot compute risk.")
        return None, 0.0
    value_per_1usd_per_lot = si.tick_value / si.tick_size  # $ per 1.0 price move for 1.0 lot

    # Risk per lot at this SL distance
    risk_per_lot = value_per_1usd_per_lot * float(sl_usd)

    if risk_per_lot <= 0:
        return None, 0.0

    raw_lot = risk_amount / risk_per_lot

    # Conform to broker lot rules
    lot = _round_to_step(raw_lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))

    return lot, risk_amount

def compute_sl_price(symbol, action_type, entry_price, sl_usd):
    """
    Convert the fixed $ distance into a price level, respecting stops level.
    """
    si = mt5.symbol_info(symbol)
    if si is None:
        return None
    digits = si.digits
    point = si.point

    if action_type == mt5.ORDER_TYPE_BUY:
        sl = entry_price - float(sl_usd)
    else:
        sl = entry_price + float(sl_usd)

    # Respect minimum stops distance if broker enforces it
    stops_pts = getattr(si, "trade_stops_level", 0)
    if stops_pts and stops_pts > 0:
        min_dist = stops_pts * point
        diff = abs(entry_price - sl)
        if diff < min_dist:
            if action_type == mt5.ORDER_TYPE_BUY:
                sl = entry_price - min_dist
            else:
                sl = entry_price + min_dist

    # round to symbol precision
    sl = round(sl, digits)
    return sl

def send_order(symbol, action_type, lot=None):
    """
    Send a buy or sell market order sized by risk and with fixed $4 SL, no TP.
    If 'lot' is provided, it will be used; otherwise we size from risk settings.
    """

    # volume validation
    if lot is not None and lot <= 0:
        print(f"[ERROR] Invalid trade volume: {lot}")
        return

    # ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol {symbol}")
        return

    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return

    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[ERROR] Trading is not allowed for {symbol}")
        return

    # get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    # 1) Compute SL price FIRST (respects broker min distance)
    sl_price = compute_sl_price(symbol, action_type, entry_price, sl_usd_distance)
    if sl_price is None:
        print("[ERROR] Could not compute SL price.")
        return

    # 2) Use the ACTUAL distance for risk sizing
    actual_distance = abs(entry_price - sl_price)

    if lot is None:
        lot, risk_amount = compute_lot_for_risk(symbol, actual_distance, risk_per_trade_pct)  # <-- pass actual_distance
        if lot is None or lot <= 0:
            print("[ERROR] Computed lot is invalid; aborting order.")
            return
        print(f"[RISK] Balance risked: ${risk_amount:.2f} | SL distance: ${actual_distance:.2f} | Lot: {lot}")

    # round lot to broker step & validate
    lot = _round_to_step(lot, si.volume_step)
    if lot < si.volume_min or lot > si.volume_max:
        print(f"[ERROR] Lot size {lot} out of range: min={si.volume_min}, max={si.volume_max}")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": entry_price,
        "deviation": slippage,
        "magic": magic_number,
        "comment": "ChandelierEntryBot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "sl": sl_price,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"[ERROR] order_send() returned None. Last error: {mt5.last_error()}")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] ORDER FAILED: {result.retcode}, {result.comment}")
    else:
        print(f"[OK] ORDER PLACED: ticket={result.order}, price={entry_price}, sl={sl_price}, lot={lot}")

        # === (post-fill SL adjust to exactly +-$10 from FILLED price) ===
        pos = get_position(symbol)  # your helper filters by magic+symbol
        if pos:
            filled = pos.price_open
            desired_sl = compute_sl_price(symbol, action_type, filled, sl_usd_distance)
            # if broker set a different SL (or none), bring it to desired_sl
            current_sl = pos.sl if pos.sl and pos.sl != 0.0 else None
            if desired_sl and (current_sl is None or abs(desired_sl - current_sl) > si.point):
                mod = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": desired_sl,
                    # no "tp" to keep no take-profit
                    "magic": magic_number,
                    "comment": "Adjust SL to filled ± $10",
                }
                mod_res = mt5.order_send(mod)
                if mod_res is not None and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[OK] SL ADJUSTED to {desired_sl}")
                else:
                    print(
                        f"[WARN] SL adjust failed: {getattr(mod_res, 'retcode', None)}, "
                        f"{getattr(mod_res, 'comment', None)}")

def close_position(position, symbol):
    """
    Close an open position (market).
    """
    if position is None:
        return
    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return
    price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": action_type,
        "position": position.ticket,
        "price": price,
        "deviation": slippage,
        "magic": magic_number,
        "comment": "Close opposite position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(close_request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {getattr(result, 'retcode', None)}, {getattr(result, 'comment', None)}")
    else:
        print(f"[OK] POSITION CLOSED: {result}")

def _make_dt(base_dt, hhmm, tz):
    h, m = map(int, hhmm.split(":"))
    return base_dt.astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0)

def _fmt_hms(total_seconds: int) -> str:
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def in_session(now_local):
    # now_local must be timezone-aware in Europe/London
    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return True
    return False

def next_session_start(now_local):
    # Returns the next session start (timezone-aware)
    candidates = []
    for start, _ in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        if start_dt > now_local:
            candidates.append(start_dt)
    if candidates:
        return min(candidates)
    # else: first window tomorrow
    tomorrow = now_local + timedelta(days=1)
    return _make_dt(tomorrow, session_windows[0][0], local_tz)

def current_session_bounds(now_local):
    """
    If we're in a session, return (start_dt, end_dt).
    Otherwise return (None, None).
    """
    for start, end in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        end_dt   = _make_dt(now_local, end,   local_tz)
        if start_dt <= now_local < end_dt:
            return start_dt, end_dt
    return None, None

def time_left_in_session(now_local):
    start_dt, end_dt = current_session_bounds(now_local)
    if end_dt is None:
        return 0
    return max(0, int((end_dt - now_local).total_seconds()))

# === HEIKIN ASHI CALCULATION === #
def calculate_heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)

    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
    ha_df['ha_low']  = pd.concat([df['low'],  ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)
    return ha_df

# === CALCULATE INDICATORS: ATR, Stops, Direction === #
def calculate_indicators(df, useHeikinAshi=use_heikin_ashi, atrPeriod=atr_period, atrMult=atr_mult):
    """
    Mirrors the TradingView Pine logic:
      - optional HA candles for calculation
      - TR like Pine
      - RMA(ATR) with SMA seed (like ta.rma)
      - long/short stops smoothed using c[1] vs previous smoothed stops
      - direction/flip & buy/sell signals using previous smoothed stops
    """
    # 1) choose candles like Pine (HA vs regular)
    if useHeikinAshi:
        ha = calculate_heikin_ashi(df)
        o = ha['ha_open']; h = ha['ha_high']; l = ha['ha_low']; c = ha['ha_close']
    else:
        o = df['open'];     h = df['high'];     l = df['low'];     c = df['close']

    tr = pd.DataFrame(index=df.index)
    tr['o'] = o; tr['h'] = h; tr['l'] = l; tr['c'] = c
    tr['c_prev'] = tr['c'].shift(1)

    # 2) true range like Pine
    def _tr(row):
        if pd.isna(row['c_prev']):
            return row['h'] - row['l']
        return max(row['h'] - row['l'], abs(row['h'] - row['c_prev']), abs(row['l'] - row['c_prev']))
    tr['true_range'] = tr.apply(_tr, axis=1)

    # 3) RMA ATR with SMA seed (like ta.rma)
    n = int(max(1, atrPeriod))
    if n == 1:
        tr['atr'] = tr['true_range']
    else:
        vals = tr['true_range'].to_numpy()
        rma = [None] * len(vals)
        if len(vals) >= n:
            sma_seed = float(pd.Series(vals[:n]).mean())
            rma[n-1] = sma_seed
            alpha = 1.0 / n
            for i in range(n, len(vals)):
                rma[i] = rma[i-1] + alpha * (vals[i] - rma[i-1])
        tr['atr'] = pd.Series(rma, index=tr.index).ffill().bfill()

    atr_val = atrMult * tr['atr']

    # 4) raw CE stops (highest/lowest over length)
    hh = tr['h'].rolling(window=n, min_periods=n).max()
    ll = tr['l'].rolling(window=n, min_periods=n).min()
    long_stop  = hh - atr_val
    short_stop = ll + atr_val

    # 5) smooth stops like Pine using c[1] vs previous smoothed value
    lss = long_stop.copy()
    sss = short_stop.copy()
    for i in range(len(tr)):
        if i == 0:
            continue
        long_prev  = lss.iloc[i-1] if pd.notna(lss.iloc[i-1]) else long_stop.iloc[i]
        short_prev = sss.iloc[i-1] if pd.notna(sss.iloc[i-1]) else short_stop.iloc[i]

        if tr['c'].iloc[i-1] > long_prev:
            lss.iloc[i] = max(long_stop.iloc[i], long_prev)
        else:
            lss.iloc[i] = long_stop.iloc[i]

        if tr['c'].iloc[i-1] < short_prev:
            sss.iloc[i] = min(short_stop.iloc[i], short_prev)
        else:
            sss.iloc[i] = short_stop.iloc[i]

    # 6) direction & flips using previous smoothed stops (dir is sticky)
    dir_vals = [1]
    for i in range(1, len(tr)):
        if tr['c'].iloc[i] > sss.iloc[i-1]:
            dir_vals.append(1)
        elif tr['c'].iloc[i] < lss.iloc[i-1]:
            dir_vals.append(-1)
        else:
            dir_vals.append(dir_vals[-1])

    tr['dir'] = dir_vals
    tr['dir_prev'] = tr['dir'].shift(1)
    tr['buy_signal']  = (tr['dir'] ==  1) & (tr['dir_prev'] == -1)
    tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] ==  1)

    # optional: expose the stops for debugging
    tr['long_stop_smooth']  = lss
    tr['short_stop_smooth'] = sss

    # expose HA-like columns to match your debug printout
    tr['ha_open'] = o
    tr['ha_high'] = h
    tr['ha_low']  = l
    tr['ha_c']    = c

    return tr

def _next_5m_close(now):
    """Return the next 5-minute bar close time (timezone-aware)."""
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    if mins_to_add == 0 and now.second == 0 and now.microsecond == 0:
        mins_to_add = 5
    return base + timedelta(minutes=mins_to_add)

def countdown_to_next_5m(tz, until_dt=None):
    """Live countdown to the next 5-min candle close, prints mm:ss.
       If until_dt is provided, never count past it. Returns the target close."""
    target = _next_5m_close(datetime.now(tz))
    if until_dt is not None and target > until_dt:
        target = until_dt

    print(f"\n[*] Waiting until next 5-min candle close at {target.strftime('%H:%M:%S %Z')} ...")
    while True:
        now = datetime.now(tz)
        remain = (target - now).total_seconds()
        if remain <= 0:
            break
        m, s = divmod(int(remain + 0.5), 60)
        # sys.stdout.write(f"\r    TIME REMAINING: {m:02d}:{s:02d}")      # Timer Countdown
        # sys.stdout.flush()
        time.sleep(0.25)
    # print("\rTIME REMAINING: 00:00")
    time.sleep(2)  # allow OANDA to finalize the bar
    return target

def countdown_to(target_dt, tz):
    print(f"\n[*] Waiting until {target_dt.strftime('%H:%M:%S %Z')} ...")
    while True:
        now = datetime.now(tz)
        remain = (target_dt - now).total_seconds()
        if remain <= 0:
            break
        m, s = divmod(int(remain + 0.5), 60)
        # sys.stdout.write(f"\r    TIME REMAINING: {m:02d}:{s:02d}")      # Timer Countdown
        # sys.stdout.flush()
        time.sleep(0.25)
    print("\rTIME REMAINING: 00:00")
    time.sleep(2)

# === MAIN TRADING LOOP === #
last_candle_time = None
last_signal = None

# session state
current_session_start = None
current_session_end = None
saw_candle_after_session_start = False

while True:
    # 1) wait until we're inside a trading session
    while True:
        now_local = datetime.now(local_tz)
        if in_session(now_local):
            # initialize session bounds/state when we enter
            current_session_start, current_session_end = current_session_bounds(now_local)
            saw_candle_after_session_start = False
            print(f"\n[+] IN SESSION: "
                  f"{current_session_start.strftime('%H:%M:%S')}–{current_session_end.strftime('%H:%M:%S')} "
                  f"{now_local.tzname()}")
            break

        nxt = next_session_start(now_local)
        print(f"\n[*] OUTSIDE SESSION. NEXT SESSION: {nxt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        while True:
            now_local = datetime.now(local_tz)
            if in_session(now_local):
                sys.stdout.write("\r[+] SESSION HAS STARTED!                         \n")
                sys.stdout.flush()
                current_session_start, current_session_end = current_session_bounds(now_local)
                saw_candle_after_session_start = False
                print(f"[+] IN SESSION: {current_session_start.strftime('%H:%M:%S')}–{current_session_end.strftime('%H:%M:%S')} {now_local.tzname()}")
                break
            remaining = int((nxt - now_local).total_seconds())
            if remaining <= 0:
                sys.stdout.write("\r[+] SESSION START REACHED. Waiting to enter... \n")
                sys.stdout.flush()
                time.sleep(1)
                continue
            # sys.stdout.write(f"\rTIME UNTIL NEXT SESSION: {_fmt_hms(remaining)} ")
            # sys.stdout.flush()
            time.sleep(1)

        # once here, we're in session; break outer while True to proceed
        if in_session(datetime.now(local_tz)):
            break

    # 2) once inside a session, wait for the next 5-minute candle close
    next_close = countdown_to_next_5m(local_tz, until_dt=current_session_end)

    # If we already have the candle that just closed (time-stamped at next_close - 5m),
    # there won't be a *new* timestamp until the *next* close. Skip the noisy retry window.
    if last_candle_time is not None and last_candle_time >= (next_close - timedelta(minutes=5)):
        print("[SKIP] Just-closed candle already fetched. Waiting to the next close for a brand-new timestamp...")
        next_next_close = next_close + timedelta(minutes=5)
        if next_next_close < current_session_end:
            countdown_to(next_next_close, local_tz)
        # else: session will end before that next close; fall through to guards

    # if we hit session end, skip trading this cycle
    if datetime.now(local_tz) >= current_session_end:
        print("[INFO] SESSION ENDED BEFORE THE NEXT BAR CLOSE. HANDLING SESSION END...")
        if auto_close_at_session_end:
            pos = get_position(mt5_symbol)
            if pos:
                print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                close_position(pos, mt5_symbol)
        continue  # go back to the top to wait for the next session

    # After the (adjusted) countdown, a new candle *should* be there.
    # Keep retries short to handle minor API latency only.
    retry_timeout = timedelta(seconds=20)
    start_time = datetime.now(local_tz)

    df = None
    prev_last = last_candle_time  # freeze the anchor for this polling phase

    # then wait for a truly new candle to be returned by OANDA
    while True:
        df = fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=num_candles)

        if df is None or df.empty:
            print("[ERROR] Failed to retrieve data from OANDA.")
            time.sleep(2)
            # timeout check (timezone-aware on both sides)
            if datetime.now(local_tz) - start_time > retry_timeout:
                print(f"[TIMEOUT] No new candle after {retry_timeout.seconds} seconds. Skipping this cycle.")
                df = None
                break
            continue

        latest_candle_time = df.index[-1]

        # bail out if session ends during retries
        if datetime.now(local_tz) >= current_session_end:
            print("[INFO] SESSION ENDED DURING DATA WAIT. HANDLING SESSION END...")
            if auto_close_at_session_end:
                pos = get_position(mt5_symbol)
                if pos:
                    print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                    close_position(pos, mt5_symbol)
            df = None
            break

        # detect truly new candle vs the anchor captured BEFORE the loop
        if prev_last is None or latest_candle_time > prev_last:
            last_candle_time = latest_candle_time  # update only when new is confirmed
            print(f"\n[OK] New 5-min candle detected: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
            break
        else:
            print(f"[WAIT] No new candle yet. Latest: {latest_candle_time.strftime('%H:%M:%S')} | Retrying in 2 seconds...")
            time.sleep(2)

        # timeout check — keep both sides timezone-aware
        if datetime.now(local_tz) - start_time > retry_timeout:
            print(f"[TIMEOUT] No new candle after {retry_timeout.seconds} seconds. Skipping this cycle.")
            df = None
            break

    # no fresh bar => skip this iteration
    if df is None or df.empty:
        continue

    # mark first post-session-start bar if required
    if require_post_start_candle and not saw_candle_after_session_start:
        if latest_candle_time >= current_session_start:
            saw_candle_after_session_start = True
        else:
            print("[INFO] Ignoring pre-session bar. Waiting for first post-session-start candle...")
            continue

    # proceed with signal generation and trading
    print(f"[OK] Retrieved {len(df)} candles from OANDA.")

    # raw candles (last 10)
    print("\n= = = = =   RAW OANDA CANDLESTICK DATA (LAST 10 CANDLES)   = = = = =")
    print(df.assign(time=df.index.strftime('%Y-%m-%d %H:%M')).set_index('time').tail(10))

    # indicators & signals
    tr = calculate_indicators(df, useHeikinAshi=use_heikin_ashi, atrPeriod=atr_period, atrMult=atr_mult)
    latest = tr.iloc[-1]

    # HA debug (last 30)
    print("\n= = = = =   LAST 10 HEIKIN-ASHI CANDLES WITH SIGNALS  = = = = =")
    debug_df = tr[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'buy_signal', 'sell_signal']].copy()
    debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
    debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
    print(debug_df[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'signal']].tail(30))

    # decide signal
    signal = None
    if latest['buy_signal']:
        signal = 'BUY'
    elif latest['sell_signal']:
        signal = 'SELL'

    # current open position
    position = get_position(mt5_symbol)
    open_position = None
    if position:
        open_position = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        print(f"[INFO] OPEN POSITION: {open_position}, volume: {position.volume}, entry: {position.price_open}")
    else:
        print("[INFO] No open position currently.")

    # final session guard before trading
    if datetime.now(local_tz) >= current_session_end:
        print("[INFO] SESSION ENDED BEFORE TRADE EXECUTION (post-calc guard).")
        if auto_close_at_session_end:
            pos = get_position(mt5_symbol)
            if pos:
                print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                close_position(pos, mt5_symbol)
        continue

    # execute according to signals
    if signal == 'BUY':
        if open_position == 'SELL':
            close_position(position, mt5_symbol)
        if open_position != 'BUY':
            print("[EXEC] BUY signal → sending order (auto-size).")
            send_order(mt5_symbol, mt5.ORDER_TYPE_BUY)

    elif signal == 'SELL':
        if open_position == 'BUY':
            close_position(position, mt5_symbol)
        if open_position != 'SELL':
            print("[EXEC] SELL signal → sending order (auto-size).")
            send_order(mt5_symbol, mt5.ORDER_TYPE_SELL)

    # end-of-iteration: if session ended now, tidy up and loop to next session
    if datetime.now(local_tz) >= current_session_end:
        print("[INFO] SESSION JUST ENDED!")
        if auto_close_at_session_end:
            pos = get_position(mt5_symbol)
            if pos:
                print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                close_position(pos, mt5_symbol)
        continue
