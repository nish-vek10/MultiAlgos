"""
Chandelier Exit Strategy for XAUUSD (Trend/Normal) - M1
Using Heikin-Ashi Candles
XAU - 1 Minute TF
"""

# ===== FILENAME: CEStrategy_XAU_1M =====

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

# Timeframe config
timeframe = mt5.TIMEFRAME_M1      # << M1
num_candles = 2000                # more history for M1

# Trading config
slippage = 10
use_heikin_ashi = True
atr_period = 1
atr_mult = 1.85
magic_number = 112233  # Unique ID for this EA's trades
local_tz = timezone('Europe/London')

# === RISK & STOPS CONFIG === #
risk_per_trade_pct = 0.25      # percent of balance risked per trade
sl_usd_distance    = 4.50      # fixed $ distance from entry (price dollars)

# === SESSION FILTER (Europe/London local time) === #
# M1 trend-friendly windows, trimmed at edges for spreads/volatility flush
session_windows = [
    ("07:10", "11:20"),
    ("13:45", "17:15"),
]

# session behaviour
auto_close_at_session_end = True       # close any open position when session ends
require_post_start_candle = True       # first trade only after the first candle formed AFTER session start

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52498279
mt5_password = "eQSX7dfh!nIZOy"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\52480967\terminal64.exe"

print("MT5 Path Exists?", os.path.exists(mt5_terminal_path))

# === CONNECT TO MT5 === #
if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server, path=mt5_terminal_path):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    sys.exit(1)

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 ACCOUNT CONNECTED! \nACCOUNT: {account.login} \nBALANCE: ${account.balance:.2f}\n")

symbol_info = mt5.symbol_info(mt5_symbol)
if symbol_info:
    print(f"[INFO] {mt5_symbol} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
    print(symbol_info._asdict())

    # Show broker minimum stop distance in price dollars (XAUUSD quote)
    min_sl_usd = (getattr(symbol_info, "trade_stops_level", 0) or 0) * symbol_info.point
    print(f"[INFO] Min stop distance enforced by broker ≈ ${min_sl_usd:.2f}")
else:
    print(f"[ERROR] Unable to fetch symbol info for {mt5_symbol}")

# === OANDA CONFIG === #
oanda_token = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_account_id = "101-004-35770497-001"
oanda_api_url = "https://api-fxpractice.oanda.com/v3"

def timeframe_to_minutes(tf):
    if tf == mt5.TIMEFRAME_M1: return 1
    if tf == mt5.TIMEFRAME_M5: return 5
    raise ValueError(f"Unsupported timeframe: {tf}")

def timeframe_to_oanda_granularity(tf):
    if tf == mt5.TIMEFRAME_M1: return "M1"
    if tf == mt5.TIMEFRAME_M5: return "M5"
    raise ValueError(f"Unsupported timeframe: {tf}")

def fetch_oanda_candles(symbol=oanda_symbol, granularity=timeframe_to_oanda_granularity(timeframe), count=num_candles):
    """
    Fetch candle data from OANDA REST API.
    symbol: OANDA instrument name (e.g., "XAU_USD")
    granularity: M1, M5, M15, H1, D, etc.
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
    Uses trade_tick_value/trade_tick_size when available; falls back safely.
    Assumes these values are for 1.0 lot (standard in MT5).
    'sl_usd' is a price distance in USD (XAUUSD quote).
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

    value_per_1usd_per_lot = None  # $ value of a $1 price move for 1 lot

    # Preferred
    tv = getattr(si, "trade_tick_value", None)
    ts = getattr(si, "trade_tick_size", None)
    if tv not in (None, 0) and ts not in (None, 0):
        value_per_1usd_per_lot = float(tv) / float(ts)

    # Fallbacks
    if value_per_1usd_per_lot is None:
        tv2 = getattr(si, "tick_value", None)
        ts2 = getattr(si, "tick_size", None)
        if tv2 not in (None, 0) and ts2 not in (None, 0):
            value_per_1usd_per_lot = float(tv2) / float(ts2)

    if value_per_1usd_per_lot is None:
        tv3 = getattr(si, "tick_value", None)
        pt  = getattr(si, "point", None)
        if tv3 not in (None, 0) and pt not in (None, 0):
            value_per_1usd_per_lot = float(tv3) / float(pt)

    if value_per_1usd_per_lot is None:
        cs = getattr(si, "trade_contract_size", None)
        if cs not in (None, 0):
            value_per_1usd_per_lot = float(cs)

    if value_per_1usd_per_lot in (None, 0):
        print("[ERROR] Cannot derive $ value per $1 move for 1 lot; missing tick fields.")
        return None, 0.0

    risk_per_lot = value_per_1usd_per_lot * float(sl_usd)
    if risk_per_lot <= 0:
        return None, 0.0

    raw_lot = risk_amount / risk_per_lot

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

    # Respect minimum stop distance
    stops_pts = getattr(si, "trade_stops_level", 0)
    if stops_pts and stops_pts > 0:
        min_dist = stops_pts * point
        diff = abs(entry_price - sl)
        if diff < min_dist:
            if action_type == mt5.ORDER_TYPE_BUY:
                sl = entry_price - min_dist
            else:
                sl = entry_price + min_dist

    return round(sl, digits)

def _next_bar_close(now, bar_minutes: int):
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (bar_minutes - (base.minute % bar_minutes)) % bar_minutes
    if mins_to_add == 0 and now.second == 0 and now.microsecond == 0:
        mins_to_add = bar_minutes
    return base + timedelta(minutes=mins_to_add)

def countdown_to_next_bar(tz, bar_minutes, until_dt=None):
    target = _next_bar_close(datetime.now(tz), bar_minutes)
    if until_dt is not None and target > until_dt:
        target = until_dt
    print(f"\n[*] Waiting until next {bar_minutes}-min candle close at {target.strftime('%H:%M:%S %Z')} ...")
    while True:
        now = datetime.now(tz)
        remain = (target - now).total_seconds()
        if remain <= 0:
            break
        time.sleep(0.25)
    time.sleep(2)  # allow OANDA to finalize the bar
    return target

def countdown_to(target_dt, tz):
    print(f"\n[*] Waiting until {target_dt.strftime('%H:%M:%S %Z')} ...")
    while True:
        now = datetime.now(tz)
        remain = (target_dt - now).total_seconds()
        if remain <= 0:
            break
        time.sleep(0.25)
    print("\rTIME REMAINING: 00:00")
    time.sleep(2)

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
    candidates = []
    for start, _ in session_windows:
        start_dt = _make_dt(now_local, start, local_tz)
        if start_dt > now_local:
            candidates.append(start_dt)
    if candidates:
        return min(candidates)
    tomorrow = now_local + timedelta(days=1)
    return _make_dt(tomorrow, session_windows[0][0], local_tz)

def current_session_bounds(now_local):
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
    Mirrors the TradingView Pine logic (with optional HA source).
    """
    if useHeikinAshi:
        ha = calculate_heikin_ashi(df)
        o = ha['ha_open']; h = ha['ha_high']; l = ha['ha_low']; c = ha['ha_close']
    else:
        o = df['open'];     h = df['high'];     l = df['low'];     c = df['close']

    tr = pd.DataFrame(index=df.index)
    tr['o'] = o; tr['h'] = h; tr['l'] = l; tr['c'] = c
    tr['c_prev'] = tr['c'].shift(1)

    def _tr(row):
        if pd.isna(row['c_prev']):
            return row['h'] - row['l']
        return max(row['h'] - row['l'], abs(row['h'] - row['c_prev']), abs(row['l'] - row['c_prev']))
    tr['true_range'] = tr.apply(_tr, axis=1)

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

    hh = tr['h'].rolling(window=n, min_periods=n).max()
    ll = tr['l'].rolling(window=n, min_periods=n).min()
    long_stop  = hh - atr_val
    short_stop = ll + atr_val

    lss = long_stop.copy()
    sss = short_stop.copy()
    for i in range(len(tr)):
        if i == 0: continue
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

    tr['long_stop_smooth']  = lss
    tr['short_stop_smooth'] = sss

    tr['ha_open'] = o
    tr['ha_high'] = h
    tr['ha_low']  = l
    tr['ha_c']    = c

    return tr

def send_order(symbol, action_type, lot=None):
    """
    Send a buy or sell market order sized by risk and with fixed sl_usd_distance SL, no TP.
    """
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

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return
    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    # Compute SL price first (respects broker min distance)
    sl_price = compute_sl_price(symbol, action_type, entry_price, sl_usd_distance)
    if sl_price is None:
        print("[ERROR] Could not compute SL price.")
        return

    # Use actual distance for risk sizing
    actual_distance = abs(entry_price - sl_price)

    if lot is None:
        lot, risk_amount = compute_lot_for_risk(symbol, actual_distance, risk_per_trade_pct)
        if lot is None or lot <= 0:
            print("[ERROR] Computed lot is invalid; aborting order.")
            return
        print(f"[RISK] Balance risked: ${risk_amount:.2f} | SL distance: ${actual_distance:.2f} | Lot: {lot}")
    else:
        if lot <= 0:
            print(f"[ERROR] Invalid trade volume: {lot}")
            return

    # Round & validate lot
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
        "comment": "ChandelierEntryBot_M1",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "sl": sl_price,
        # no TP
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"[ERROR] order_send() returned None. Last error: {mt5.last_error()}")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] ORDER FAILED: {result.retcode}, {result.comment}")
    else:
        print(f"[OK] ORDER PLACED: ticket={result.order}, price={entry_price}, sl={sl_price}, lot={lot}")

        # Post-fill SL adjust to exactly ±sl_usd_distance from FILLED price (in case broker shifted it)
        pos = get_position(symbol)
        if pos:
            filled = pos.price_open
            desired_sl = compute_sl_price(symbol, action_type, filled, sl_usd_distance)
            current_sl = pos.sl if pos.sl and pos.sl != 0.0 else None
            if desired_sl and (current_sl is None or abs(desired_sl - current_sl) > si.point):
                mod = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": desired_sl,
                    "magic": magic_number,
                    "comment": f"Adjust SL to filled ± ${sl_usd_distance}",
                }
                mod_res = mt5.order_send(mod)
                if mod_res is not None and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[OK] SL ADJUSTED to {desired_sl}")
                else:
                    print(f"[WARN] SL adjust failed: {getattr(mod_res, 'retcode', None)}, {getattr(mod_res, 'comment', None)}")

def close_position(position, symbol):
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

# === MAIN TRADING LOOP === #
last_candle_time = None

# session state
current_session_start = None
current_session_end = None
saw_candle_after_session_start = False

bar_minutes = timeframe_to_minutes(timeframe)
retry_timeout = timedelta(seconds=10)   # tighter for M1

while True:
    # 1) wait until we're inside a trading session
    while True:
        now_local = datetime.now(local_tz)
        if in_session(now_local):
            current_session_start, current_session_end = current_session_bounds(now_local)
            saw_candle_after_session_start = False
            print(f"\n[+] IN SESSION: {current_session_start.strftime('%H:%M:%S')}–{current_session_end.strftime('%H:%M:%S')} {now_local.tzname()}")
            break

        nxt = next_session_start(now_local)
        print(f"\n[*] OUTSIDE SESSION. NEXT SESSION: {nxt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        while True:
            now_local = datetime.now(local_tz)
            if in_session(now_local):
                sys.stdout.write("\r[+] SESSION HAS STARTED!\n")
                sys.stdout.flush()
                current_session_start, current_session_end = current_session_bounds(now_local)
                saw_candle_after_session_start = False
                print(f"[+] IN SESSION: {current_session_start.strftime('%H:%M:%S')}–{current_session_end.strftime('%H:%M:%S')} {now_local.tzname()}")
                break
            remaining = int((nxt - now_local).total_seconds())
            if remaining <= 0:
                sys.stdout.write("\r[+] SESSION START REACHED. Waiting to enter...\n")
                sys.stdout.flush()
                time.sleep(1)
                continue
            time.sleep(1)

        if in_session(datetime.now(local_tz)):
            break

    # 2) once inside a session, wait for the next bar close
    next_close = countdown_to_next_bar(local_tz, bar_minutes, until_dt=current_session_end)

    # Skip if we already have this just-closed bar
    if last_candle_time is not None and last_candle_time >= (next_close - timedelta(minutes=bar_minutes)):
        print("[SKIP] Just-closed candle already fetched. Waiting to the next close for a brand-new timestamp...")
        next_next_close = next_close + timedelta(minutes=bar_minutes)
        if next_next_close < current_session_end:
            countdown_to(next_next_close, local_tz)

    if datetime.now(local_tz) >= current_session_end:
        print("[INFO] SESSION ENDED BEFORE THE NEXT BAR CLOSE. HANDLING SESSION END...")
        if auto_close_at_session_end:
            pos = get_position(mt5_symbol)
            if pos:
                print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                close_position(pos, mt5_symbol)
        continue

    # 3) Fetch a truly new candle
    start_time = datetime.now(local_tz)
    df = None
    prev_last = last_candle_time

    while True:
        df = fetch_oanda_candles(symbol=oanda_symbol,
                                 granularity=timeframe_to_oanda_granularity(timeframe),
                                 count=num_candles)

        if df is None or df.empty:
            print("[ERROR] Failed to retrieve data from OANDA.")
            time.sleep(1)
            if datetime.now(local_tz) - start_time > retry_timeout:
                print(f"[TIMEOUT] No new candle after {retry_timeout.seconds} seconds. Skipping this cycle.")
                df = None
                break
            continue

        latest_candle_time = df.index[-1]

        if datetime.now(local_tz) >= current_session_end:
            print("[INFO] SESSION ENDED DURING DATA WAIT. HANDLING SESSION END...")
            if auto_close_at_session_end:
                pos = get_position(mt5_symbol)
                if pos:
                    print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                    close_position(pos, mt5_symbol)
            df = None
            break

        if prev_last is None or latest_candle_time > prev_last:
            last_candle_time = latest_candle_time
            print(f"\n[OK] New {bar_minutes}-min candle detected: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
            break
        else:
            print(f"[WAIT] No new candle yet. Latest: {latest_candle_time.strftime('%H:%M:%S')} | Retrying in 1 second...")
            time.sleep(1)

        if datetime.now(local_tz) - start_time > retry_timeout:
            print(f"[TIMEOUT] No new candle after {retry_timeout.seconds} seconds. Skipping this cycle.")
            df = None
            break

    if df is None or df.empty:
        continue

    # 4) first post-session-start bar guard
    if require_post_start_candle and not saw_candle_after_session_start:
        if last_candle_time >= current_session_start:
            saw_candle_after_session_start = True
        else:
            print("[INFO] Ignoring pre-session bar. Waiting for first post-session-start candle...")
            continue

    print(f"[OK] Retrieved {len(df)} candles from OANDA.")

    # Debug: raw candles (last 10)
    print("\n= = = RAW OANDA CANDLESTICK DATA (LAST 10) = = =")
    print(df.assign(time=df.index.strftime('%Y-%m-%d %H:%M')).set_index('time').tail(10))

    # 5) indicators & signals
    tr = calculate_indicators(df, useHeikinAshi=use_heikin_ashi, atrPeriod=atr_period, atrMult=atr_mult)
    latest = tr.iloc[-1]

    print("\n= = = LAST 10 HEIKIN-ASHI CANDLES WITH SIGNALS = = =")
    debug_df = tr[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'buy_signal', 'sell_signal']].copy()
    debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
    debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
    print(debug_df[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'signal']].tail(10))

    signal = None
    if latest['buy_signal']:
        signal = 'BUY'
    elif latest['sell_signal']:
        signal = 'SELL'

    # 6) current open position
    position = get_position(mt5_symbol)
    open_position = None
    if position:
        open_position = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        print(f"[INFO] OPEN POSITION: {open_position}, volume: {position.volume}, entry: {position.price_open}")
    else:
        print("[INFO] No open position currently.")

    # final session guard
    if datetime.now(local_tz) >= current_session_end:
        print("[INFO] SESSION ENDED BEFORE TRADE EXECUTION (post-calc guard).")
        if auto_close_at_session_end:
            pos = get_position(mt5_symbol)
            if pos:
                print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                close_position(pos, mt5_symbol)
        continue

    # 7) execute according to signals
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

    # 8) end-of-iteration: if session ended now, tidy up and loop to next session
    if datetime.now(local_tz) >= current_session_end:
        print("[INFO] SESSION JUST ENDED!")
        if auto_close_at_session_end:
            pos = get_position(mt5_symbol)
            if pos:
                print("[ACTION] CLOSING OPEN POSITION AT SESSION END...")
                close_position(pos, mt5_symbol)
        continue
