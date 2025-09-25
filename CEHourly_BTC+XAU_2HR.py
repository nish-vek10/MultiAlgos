# ===== FILENAME: CE-2H_BTC_XAU_SPX.py =====
"""
Chandelier-Direction, Fixed SL/TP — Two-Hour (H2) for XAUUSD, BTCUSD, US500 (SPX)
=================================================================================

OVERVIEW
--------
Same logic and logs as your H2 variant. This version:
- Trades XAUUSD, BTCUSD, and US500 (SPX) on H2
- No SL/TP (market entries only) for all three lanes as requested
- Fixed lots (0.1) and risk_usd disabled for these lanes
- Keeps hedging-style, per-lane magic numbers, same indicators and logging

NOTE: Keep credentials out of source in production.
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
import os
import sys
import requests
import pandas as pd
from pytz import timezone
from utils.oanda_fetch import make_oanda_session, fetch_candles

# === CONFIGURATION === #
timeframe_default = mt5.TIMEFRAME_H1  # not used; OANDA granularity drives TF
num_candles = 750
lot_size_default = 0.1
slippage = 10
atr_period = 1
atr_mult = 1.85
local_tz = timezone('Europe/London')

# --- Logging verbosity ---
VERBOSE_LOG = True
PRINT_LAST_N = 20

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52512991
mt5_password = "s8yQu6pE5!grDc"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\52512991\terminal64.exe"

print("MT5 Path Exists?", os.path.exists(mt5_terminal_path))

# === CONNECT TO MT5 === #
if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server, path=mt5_terminal_path):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    sys.exit(1)

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 Account Connected! \nAccount: {account.login} \nBalance: {account.balance:.2f} {account.currency}\n")

# === OANDA DEMO (DATA SOURCE) === #
oanda_token = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_account_id = "101-004-35770497-001"
oanda_api_url = "https://api-fxpractice.oanda.com/v3"
oanda_sess = make_oanda_session(oanda_token, host="https://api-fxpractice.oanda.com", timeout=(3.05, 10))

# -------------------------------------------------------------------------------------------------
# STRATEGY MATRIX — H2 only, no SL/TP, fixed lots 0.1
# Unique magic numbers per lane
# -------------------------------------------------------------------------------------------------
strategies = [
    {"id": "XAU-H2", "mt5": "XAUUSD", "oanda": "XAU_USD",    "granularity": "H2",
     "fixed_sl": None, "fixed_tp": None, "lot_size": 0.5, "risk_usd": None, "magic": 98765421},

    {"id": "BTC-H2", "mt5": "BTCUSD", "oanda": "BTC_USD",    "granularity": "H2",
     "fixed_sl": None, "fixed_tp": None, "lot_size": 0.5, "risk_usd": None, "magic": 98765422},

    {"id": "SPX-H2", "mt5": "US500",  "oanda": "SPX500_USD", "granularity": "H2",
     "fixed_sl": None, "fixed_tp": None, "lot_size": 25.0, "risk_usd": None, "magic": 98765423},

    {"id": "EURUSD-H2", "mt5": "EURUSD",  "oanda": "EUR_USD", "granularity": "H2",
     "fixed_sl": None, "fixed_tp": None, "lot_size": 0.5, "risk_usd": None, "magic": 98765424},

    {"id": "EURUSD-H2", "mt5": "USDJPY", "oanda": "USD_JPY", "granularity": "H2",
     "fixed_sl": None, "fixed_tp": None, "lot_size": 1.0, "risk_usd": None, "magic": 98765425},
]

# =================================================================================================
# ====== Utility: Symbol tick/money specs and risk calculators
# =================================================================================================
def get_tick_specs(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return (None, None, None, None)

    t_size = getattr(info, "trade_tick_size", None)
    t_val  = getattr(info, "trade_tick_value", None)
    if t_size and t_val:
        return (t_size, t_val, "account", getattr(info, "currency_profit", None))

    t_size = getattr(info, "tick_size", None) or getattr(info, "point", None)
    t_val  = getattr(info, "tick_value", None)
    return (t_size, t_val, "profit", getattr(info, "currency_profit", None))

def convert_to_account_currency(amount, from_ccy, account_ccy="USD"):
    if amount is None:
        return None
    if not from_ccy or from_ccy == account_ccy:
        return amount

    def mid(symbol):
        if mt5.symbol_select(symbol, True):
            t = mt5.symbol_info_tick(symbol)
            if t:
                return (t.bid + t.ask) / 2.0
        return None

    direct = f"{from_ccy}{account_ccy}"
    m = mid(direct)
    if m:
        return amount * m

    inv = f"{account_ccy}{from_ccy}"
    m = mid(inv)
    if m:
        return amount / m

    if from_ccy != "USD" and account_ccy != "USD":
        a = convert_to_account_currency(amount, from_ccy, "USD")
        return convert_to_account_currency(a, "USD", account_ccy) if a is not None else None
    return None

def lots_for_target_usd(symbol, sl_distance, target_usd, account_ccy="USD"):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    if sl_distance is None:
        return None  # risk-based sizing requires SL distance

    tick_size, tick_value, unit, profit_ccy = get_tick_specs(symbol)
    if not tick_size or not tick_value or tick_size == 0:
        return None

    ticks = sl_distance / tick_size
    risk_per_lot = ticks * float(tick_value)

    if unit == "profit":
        risk_per_lot_acct = convert_to_account_currency(risk_per_lot, profit_ccy, account_ccy)
    else:
        risk_per_lot_acct = risk_per_lot

    if not risk_per_lot_acct or risk_per_lot_acct <= 0:
        return None

    raw_lots = float(target_usd) / risk_per_lot_acct

    step = info.volume_step or 0.01
    raw_lots = round(raw_lots / step) * step
    raw_lots = max(raw_lots, info.volume_min)
    raw_lots = min(raw_lots, info.volume_max)

    step_str = f"{step:.10f}".rstrip("0")
    dec = len(step_str.split(".")[1]) if "." in step_str else 0
    return float(f"{raw_lots:.{dec}f}")

def estimate_risk(symbol, sl_distance, lots, account_ccy="USD"):
    info = mt5.symbol_info(symbol)
    if info is None or sl_distance is None:
        return (None, None, None)

    tick_size, tick_value, unit, profit_ccy = get_tick_specs(symbol)
    if not tick_size or not tick_value or tick_size == 0:
        return (None, None, None)

    ticks = sl_distance / tick_size
    risk_1lot = ticks * float(tick_value)

    if unit == "profit":
        risk_1lot_acct = convert_to_account_currency(risk_1lot, profit_ccy, account_ccy)
    else:
        risk_1lot_acct = risk_1lot

    if not risk_1lot_acct:
        return (None, profit_ccy, None)

    risk_acct = risk_1lot_acct * float(lots)
    risk_profit = None if unit == "account" else (risk_1lot * float(lots))
    return (risk_acct, profit_ccy, risk_profit)

# =================================================================================================
# ====== Data: OANDA candles
# =================================================================================================
def fetch_oanda_candles(symbol, granularity="H2", count=500):
    try:
        j = fetch_candles(oanda_sess, instrument=symbol, granularity=granularity, count=count, price="M")
    except Exception as e:
        msg = str(e)
        print(f"[ERROR] OANDA {symbol}/{granularity}: {msg[:600]}")
        return None

    raw_candles = j.get("candles", [])
    if not raw_candles:
        return None

    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for c in raw_candles:
        if c.get("complete"):
            utc_time = pd.to_datetime(c["time"], utc=True)
            local_time = utc_time.tz_convert(local_tz)
            data["time"].append(local_time)
            data["open"].append(float(c["mid"]["o"]))
            data["high"].append(float(c["mid"]["h"]))
            data["low"].append(float(c["mid"]["l"]))
            data["close"].append(float(c["mid"]["c"]))
            data["volume"].append(int(c["volume"]))

    if not data["time"]:
        return None

    df = pd.DataFrame(data).set_index("time")
    return df

# =================================================================================================
# ====== Trading helpers (MT5)
# =================================================================================================
def get_position(symbol, magic=None):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    if magic is None:
        return positions[0]
    for p in positions:
        if getattr(p, "magic", None) == magic:
            return p
    return None

def send_order(symbol, action_type, lot, sl_distance, tp_distance=None, magic=None, comment="ChandelierEntryBot"):
    """
    Submit a market order with optional SL/TP. If sl_distance/tp_distance is None, omit that field.
    """
    if lot is None or lot <= 0:
        print(f"[ERROR] Invalid trade volume: {lot}")
        return

    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol {symbol}")
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return

    if hasattr(symbol_info, "trade_allowed") and not symbol_info.trade_allowed:
        print(f"[ERROR] Trading is not allowed for {symbol}")
        return

    step = symbol_info.volume_step or 0.01
    lot = round(lot / step) * step
    if lot < symbol_info.volume_min or lot > symbol_info.volume_max:
        print(f"[ERROR] Lot size {lot} out of range: min={symbol_info.volume_min}, max={symbol_info.volume_max}")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return

    price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    # Only compute/include SL/TP when distances provided
    sl_price = None
    tp_price = None
    if sl_distance is not None:
        sl_price = price - sl_distance if action_type == mt5.ORDER_TYPE_BUY else price + sl_distance
    if tp_distance is not None:
        tp_price = price + tp_distance if action_type == mt5.ORDER_TYPE_BUY else price - tp_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": price,
        "deviation": slippage,
        "magic": magic if magic is not None else 987654,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if sl_price is not None:
        request["sl"] = sl_price
    if tp_price is not None:
        request["tp"] = tp_price

    result = mt5.order_send(request)
    if result is None:
        print(f"[ERROR] No response from order_send for {symbol}")
        return False
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order failed on {symbol}: Retcode={result.retcode}, Comment={result.comment}")
        return False
    else:
        print(f"[OK] Order placed on {symbol}: Ticket={result.order} | Comment={comment}")
        return True


def close_position(position, symbol):
    if position is None:
        return
    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Could not get tick to close {symbol}")
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
        "magic": getattr(position, "magic", 0),
        "comment": "Close opposite position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(close_request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {getattr(result, 'retcode', None)}, {getattr(result, 'comment', None)}")
    else:
        print(f"[OK] Position closed: {result.order}")

# =================================================================================================
# ====== Indicators (unchanged)
# =================================================================================================
def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)

    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
    ha_df['ha_low']  = pd.concat([df['low'],  ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)
    return ha_df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ha_df = calculate_heikin_ashi(df)

    tr = pd.DataFrame(index=ha_df.index)
    tr['ha_h'] = ha_df['ha_high']
    tr['ha_l'] = ha_df['ha_low']
    tr['ha_c'] = ha_df['ha_close']
    tr['prev_ha_c'] = tr['ha_c'].shift(1)

    tr['ha_open'] = ha_df['ha_open']
    tr['ha_high'] = ha_df['ha_high']
    tr['ha_low']  = ha_df['ha_low']

    tr['true_range'] = tr[['ha_h', 'ha_l', 'prev_ha_c']].apply(
        lambda row: max(
            row['ha_h'] - row['ha_l'],
            abs(row['ha_h'] - row['prev_ha_c']) if pd.notna(row['prev_ha_c']) else 0.0,
            abs(row['ha_l'] - row['prev_ha_c']) if pd.notna(row['prev_ha_c']) else 0.0
        ), axis=1
    )
    tr['atr'] = tr['true_range'].ewm(alpha=1/atr_period, adjust=False).mean()

    long_stop  = tr['ha_h'].rolling(window=atr_period).max() - (tr['atr'] * atr_mult)
    short_stop = tr['ha_l'].rolling(window=atr_period).min() + (tr['atr'] * atr_mult)

    long_s = long_stop.copy()
    short_s = short_stop.copy()

    for i in range(1, len(tr)):
        if tr['ha_c'].iloc[i - 1] > long_s.iloc[i - 1]:
            long_s.iloc[i] = max(long_stop.iloc[i], long_s.iloc[i - 1])
        else:
            long_s.iloc[i] = long_stop.iloc[i]
        if tr['ha_c'].iloc[i - 1] < short_s.iloc[i - 1]:
            short_s.iloc[i] = min(short_stop.iloc[i], short_s.iloc[i - 1])
        else:
            short_s.iloc[i] = short_stop.iloc[i]

    direction = [1]
    for i in range(1, len(tr)):
        if tr['ha_c'].iloc[i] > short_s.iloc[i - 1]:
            direction.append(1)
        elif tr['ha_c'].iloc[i] < long_s.iloc[i - 1]:
            direction.append(-1)
        else:
            direction.append(direction[-1])

    tr['dir'] = direction
    tr['dir_prev'] = tr['dir'].shift(1)
    tr['buy_signal'] = (tr['dir'] == 1) & (tr['dir_prev'] == -1)
    tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] == 1)
    return tr

def _attempt_execution_for_signal(s, signal, m_sym, position, account_currency):
    """
    Try to realize the desired 'signal' on symbol m_sym.
    Returns True if we ended up in the desired side or placed the order successfully.
    """
    open_pos = None
    if position:
        open_pos = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

    # If we already have the correct side, nothing to do.
    if open_pos == signal:
        return True

    # If we hold opposite, close it first.
    if open_pos and open_pos != signal:
        close_position(position, m_sym)
        # refresh position snapshot after close
        time.sleep(0.5)
        position = get_position(m_sym, magic=s['magic'])
        open_pos = 'BUY' if (position and position.type == mt5.ORDER_TYPE_BUY) else ('SELL' if position else None)

    # If not already in desired side, place order
    if open_pos != signal:
        use_lot = s.get('lot_size') or 0.1
        # risk_usd only when fixed_sl is defined (your current logic)
        if s.get('risk_usd') is not None and s.get('fixed_sl') is not None:
            dyn = lots_for_target_usd(m_sym, s['fixed_sl'], s['risk_usd'], account_currency)
            use_lot = dyn if dyn is not None else use_lot

        order_type = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL
        ok = send_order(m_sym, order_type, use_lot, s['fixed_sl'], s['fixed_tp'],
                        magic=s['magic'], comment=f"CEB:{s['id']}")
        # If order_send said OK, we consider it executed.
        if ok:
            return True

    return False


# =================================================================================================
# ====== Startup: show symbol info and risk examples
# =================================================================================================
_printed = set()
for s in strategies:
    sym = s["mt5"]
    if sym in _printed:
        continue
    _printed.add(sym)

    info = mt5.symbol_info(sym)
    if info:
        print(f"[INFO] {sym} Lot Range: min={info.volume_min}, max={info.volume_max}, step={info.volume_step}")
    else:
        print(f"[ERROR] Unable to fetch symbol info for {sym}")

for s in strategies:
    sym = s['mt5']
    lot = s.get('lot_size') or lot_size_default
    if s.get('risk_usd') is not None and s.get('fixed_sl') is not None:
        dyn = lots_for_target_usd(sym, s['fixed_sl'], s['risk_usd'], account.currency)
        lot = dyn if dyn is not None else lot

    # Only estimate risk if SL is defined
    if s.get('fixed_sl') is not None:
        r_acct, p_ccy, r_profit = estimate_risk(sym, s['fixed_sl'], lot, account.currency)
        if r_acct is not None:
            extra = f" (≈ {r_profit:.2f} {p_ccy})" if r_profit is not None else ""
            print(f"[RISK] {s['id']}: lot={lot} | SL={s['fixed_sl']} | TP={s['fixed_tp']} -> est ≈ {r_acct:.2f} {account.currency}{extra}")
        else:
            print(f"[RISK] {s['id']}: lot={lot} | SL={s['fixed_sl']} | TP={s['fixed_tp']} -> est risk unavailable")
    else:
        print(f"[RISK] {s['id']}: lot={lot} | SL=None | TP=None (no SL/TP)")

# =================================================================================================
# ====== Per-strategy state
# =================================================================================================
last_candle_time = {s['id']: None for s in strategies}
last_signal      = {s['id']: None for s in strategies}
last_signal_info = {s['id']: None for s in strategies}

pending_signal   = {s['id']: None for s in strategies}   # 'BUY'/'SELL' we still need to execute
pending_since    = {s['id']: None for s in strategies}   # datetime when we first queued it
last_retry_at    = {s['id']: None for s in strategies}   # throttle retries
RETRY_EVERY_SECS = 20                                    # align with your loop sleep

# =================================================================================================
# ====== MAIN HEARTBEAT LOOP
# =================================================================================================
print("\n[Engine] Multi-timeframe mode: watching for new H2 candles...\n")
while True:
    for s in strategies:
        sid   = s['id']
        o_sym = s['oanda']
        m_sym = s['mt5']
        tf    = s['granularity']  # "H2"

        df = fetch_oanda_candles(symbol=o_sym, granularity=tf, count=num_candles)
        if df is None or df.empty:
            print(f"[ERROR] {sid}: no data")
            continue

        latest_candle_time = df.index[-1]
        if last_candle_time[sid] is not None and latest_candle_time <= last_candle_time[sid]:
            continue
        last_candle_time[sid] = latest_candle_time
        print(f"[OK] {sid}: new {tf} candle @ {latest_candle_time.strftime('%Y-%m-%d %H:%M')}")

        tr = calculate_indicators(df)
        latest = tr.iloc[-1]

        # --- Current and previous-bar signals ---
        signal = 'BUY' if latest['buy_signal'] else ('SELL' if latest['sell_signal'] else None)

        prev_signal = None
        if len(tr) >= 2:
            prev = tr.iloc[-2]
            if prev['buy_signal']:
                prev_signal = 'BUY'
            elif prev['sell_signal']:
                prev_signal = 'SELL'

        if VERBOSE_LOG:
            raw_tail = df[['open', 'high', 'low', 'close', 'volume']].copy().tail(PRINT_LAST_N)
            raw_tail.index = raw_tail.index.strftime('%Y-%m-%d %H:%M')
            print(f"\n===== RAW OANDA CANDLESTICK ({sid}, {tf}) — last {PRINT_LAST_N} =====")
            print(raw_tail)

            dbg = tr[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'buy_signal', 'sell_signal']].copy()
            dbg['signal'] = dbg.apply(lambda r: 'BUY' if r['buy_signal'] else ('SELL' if r['sell_signal'] else ''), axis=1)
            dbg.index = dbg.index.strftime('%Y-%m-%d %H:%M')
            print(f"\n===== HEIKIN-ASHI + SIGNALS ({sid}, {tf}) — last {PRINT_LAST_N} =====")
            print(dbg[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'signal']].tail(PRINT_LAST_N))
            print()

        signal = 'BUY' if latest['buy_signal'] else ('SELL' if latest['sell_signal'] else None)

        position = get_position(m_sym, magic=s['magic'])
        open_pos = None
        if position:
            open_pos = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

        # Snapshot current position for this magic
        position = get_position(m_sym, magic=s['magic'])
        open_pos = None
        if position:
            open_pos = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

        # === 5A) If we have a fresh signal on the latest bar, act as before ===
        if signal:
            if signal != last_signal[sid] or open_pos != signal:
                prev_sig = last_signal[sid] if last_signal[sid] else "NONE"
                print(f"[TRADE] {sid}: new signal={signal} | prev={prev_sig} | open={open_pos or 'NONE'}")

                # Record last signal time/info exactly as you do today
                last_signal[sid] = signal
                last_signal_info[sid] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {signal}"

                # Try to realize this signal immediately
                ok = _attempt_execution_for_signal(s, signal, m_sym, position, account.currency)
                if ok:
                    # Clear any pending for this lane
                    pending_signal[sid] = None
                    pending_since[sid] = None
                    last_retry_at[sid] = None
                else:
                    # Queue it as pending to auto-retry next loops
                    if not pending_signal[sid]:
                        pending_signal[sid] = signal
                        pending_since[sid] = datetime.now()
                        print(f"[PENDING] {sid}: queued {signal} execution (will retry).")
            else:
                print(f"[INFO] {sid}: signal unchanged ({signal}).")

        # === 5B) No fresh latest-bar signal: try backfilling previous-bar signal if missed ===
        else:
            if prev_signal and (last_signal[sid] != prev_signal or open_pos != prev_signal):
                # Only backfill if we haven't already queued/handled it
                if not pending_signal[sid]:
                    print(f"[BACKFILL] {sid}: previous bar had {prev_signal} — attempting execution now.")
                    ok = _attempt_execution_for_signal(s, prev_signal, m_sym, position, account.currency)
                    if ok:
                        last_signal[sid] = prev_signal
                        last_signal_info[sid] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {prev_signal}"
                        pending_signal[sid] = None
                        pending_since[sid] = None
                        last_retry_at[sid] = None
                    else:
                        pending_signal[sid] = prev_signal
                        pending_since[sid] = datetime.now()
                        print(f"[PENDING] {sid}: queued {prev_signal} from previous bar (will retry).")
            else:
                print(f"[INFO] {sid}: no actionable signal.")

        # === 5C) If we still have something pending, keep retrying until it sticks ===
        if pending_signal[sid]:
            # Re-snapshot in case something changed this loop
            position = get_position(m_sym, magic=s['magic'])
            open_pos = 'BUY' if (position and position.type == mt5.ORDER_TYPE_BUY) else ('SELL' if position else None)

            # If an opposite live signal appears now, switch/cancel pending
            if signal and signal != pending_signal[sid]:
                print(f"[CANCELLED] {sid}: live signal {signal} overrides pending {pending_signal[sid]}.")
                # Attempt the new signal right away
                ok = _attempt_execution_for_signal(s, signal, m_sym, position, account.currency)
                last_signal[sid] = signal
                last_signal_info[sid] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {signal}"
                if ok:
                    pending_signal[sid] = None
                    pending_since[sid] = None
                    last_retry_at[sid] = None
                else:
                    pending_signal[sid] = signal
                    if pending_since[sid] is None:
                        pending_since[sid] = datetime.now()
                # done handling override for this loop
            else:
                # Throttle retries roughly to the main loop cadence
                now_ts = datetime.now()
                if last_retry_at[sid] is None or (now_ts - (last_retry_at[sid])).total_seconds() >= RETRY_EVERY_SECS:
                    print(f"[RETRY] {sid}: attempting pending {pending_signal[sid]} execution...")
                    ok = _attempt_execution_for_signal(s, pending_signal[sid], m_sym, position, account.currency)
                    last_retry_at[sid] = now_ts
                    if ok:
                        print(f"[OK] {sid}: pending {pending_signal[sid]} executed.")
                        pending_signal[sid] = None
                        pending_since[sid] = None
                        last_retry_at[sid] = None

    # H2 candles every 120 min; 20s polling is fine.
    time.sleep(20)
