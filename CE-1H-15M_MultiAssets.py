# ===== FILENAME: CE-1H-15M_MultiAssets.py =====
"""
Chandelier-Direction, Fixed SL/TP — Multi-Timeframe (H1 + M15) for XAUUSD, BTCUSD, EURO STOXX 50, USDJPY, DE40
===============================================================================================================

OVERVIEW
--------
This bot runs MULTIPLE INDEPENDENT STRATEGIES per (symbol, timeframe) on the SAME ACCOUNT. It uses a
Heikin-Ashi based direction flip with Chandelier-style stops (for direction only), and submits MARKET ORDERS with
PER-SYMBOL FIXED SL & TP. Position size can be:
  - fixed lots (per strategy), or
  - dynamically sized to target a fixed risk in account currency (via `risk_usd`).

Key points:
- HEDGING-STYLE INDEPENDENCE: H1 and M15 TRADE INDEPENDENTLY and can be long and short at the same time.
  They do NOT have to agree, and DO NOT CLASH: each strategy has its OWN MAGIC NUMBER and comment tag.
- OANDA DATA drives signals (Heikin-Ashi + Chandelier-style stops). MT5 is used for trading.
- PER-STRATEGY STATE tracks last signal and last completed candle time, so each timeframe acts only on NEW CANDLES.
- RISK ACCOUNTING is computed using symbol tick specs and, if needed, FX conversion into the account currency.

NEW IMPLEMENT:
- M15 XAUUSD with $20 SL/TP, fixed lots (default 0.1 unless changed).
- M15 BTCUSD with $2,000 SL/TP, fixed lots (default 0.1 unless changed).
Existing H1 assets/settings remain unchanged.

NOTE
----
- If MT5 account is NETTING, the broker maintains a single net position per symbol. Magic numbers
  still help with bookkeeping, but opposite signals will OFFSET/FLIP the net exposure. For TRUE HEDGING (long & short
  simultaneously), account/broker must support HEDGING MODE. This script is designed for hedging independence.

- Keep credentials out of source in production (use environment variables).
  They are inline here for continuity with current demo setup.
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
import os
import sys
import requests
import pandas as pd
from pytz import timezone

# === CONFIGURATION === #
# Global defaults (can be overridden per-strategy below)
timeframe_default = mt5.TIMEFRAME_H1  # kept for reference; not used in logic (we drive TF via OANDA granularity)
num_candles = 750
lot_size_default = 0.1  # fallback only; overridden per strategy if provided
slippage = 10
atr_period = 1
atr_mult = 1.85
local_tz = timezone('Europe/London')

# --- Logging verbosity ---
VERBOSE_LOG = True      # set False to silence the detailed tables
PRINT_LAST_N = 20       # how many candles to print in the detailed tables

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52498249
mt5_password = "cg!rr!v26Juuh3"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\52461477\terminal64.exe"

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

# -------------------------------------------------------------------------------------------------
# STRATEGY MATRIX (each entry is an independent trading "lane" with its own magic/comment/state)
# -------------------------------------------------------------------------------------------------
# Existing H1 lanes preserved; add requested M15 lanes for XAUUSD and BTCUSD.
strategies = [
    # --- H1 (existing behavior) ---
    {"id": "XAU-H1",    "mt5": "XAUUSD",  "oanda": "XAU_USD",  "granularity": "H1",
     "fixed_sl": 40,    "fixed_tp": 40,   "lot_size": None,     "risk_usd": 400.0, "magic": 98765401},
    {"id": "BTC-H1",    "mt5": "BTCUSD",  "oanda": "BTC_USD",  "granularity": "H1",
     "fixed_sl": 4000,  "fixed_tp": 4000, "lot_size": None,     "risk_usd": 400.0, "magic": 98765402},
    {"id": "EU50-H1",   "mt5": "STOXX50", "oanda": "EU50_EUR", "granularity": "H1",
     "fixed_sl": 50,    "fixed_tp": 50,   "lot_size": None,    "risk_usd": 400.0,"magic": 98765403},
    {"id": "USDJPY-H1", "mt5": "USDJPY",  "oanda": "USD_JPY",  "granularity": "H1",
     "fixed_sl": 2,     "fixed_tp": 2,    "lot_size": None,    "risk_usd": 400.0,"magic": 98765404},
    {"id": "DE40-H1",   "mt5": "DE40",    "oanda": "DE30_EUR", "granularity": "H1",
     "fixed_sl": 250,   "fixed_tp": 250,  "lot_size": None,    "risk_usd": 400.0,"magic": 98765405},

    # --- NEW: M15 (independent, hedge-style) ---
    {"id": "XAU-M15",   "mt5": "XAUUSD",  "oanda": "XAU_USD",  "granularity": "M15",
     "fixed_sl": 10,    "fixed_tp": 40,   "lot_size": None,     "risk_usd": 400.0, "magic": 98765415},
    {"id": "BTC-M15",   "mt5": "BTCUSD",  "oanda": "BTC_USD",  "granularity": "M15",
     "fixed_sl": 500,  "fixed_tp": 1750, "lot_size": None,     "risk_usd": 400.0, "magic": 98765416},
]

# =================================================================================================
# ====== Utility: Symbol tick/money specs and risk calculators
# =================================================================================================
def get_tick_specs(symbol):
    """
    Return (tick_size, tick_value, unit, profit_ccy) for 1.0 lot.
    unit = "account"  -> tick_value is already in account currency (trade_* fields).
    unit = "profit"   -> tick_value is in profit currency (generic fields), needs FX conversion.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return (None, None, None, None)

    # Prefer trade_* (usually in account currency)
    t_size = getattr(info, "trade_tick_size", None)
    t_val  = getattr(info, "trade_tick_value", None)
    if t_size and t_val:
        return (t_size, t_val, "account", getattr(info, "currency_profit", None))

    # Fallback to generic (usually profit currency)
    t_size = getattr(info, "tick_size", None) or getattr(info, "point", None)
    t_val  = getattr(info, "tick_value", None)
    return (t_size, t_val, "profit", getattr(info, "currency_profit", None))

def convert_to_account_currency(amount, from_ccy, account_ccy="USD"):
    """
    Convert 'amount' from 'from_ccy' to 'account_ccy' using MT5 quotes (mid).
    Tries direct, inverse, then bridges via USD.
    """
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

    # Bridge via USD
    if from_ccy != "USD" and account_ccy != "USD":
        a = convert_to_account_currency(amount, from_ccy, "USD")
        return convert_to_account_currency(a, "USD", account_ccy) if a is not None else None

    return None

def lots_for_target_usd(symbol, sl_distance, target_usd, account_ccy="USD"):
    """
    Compute lots so risk ≈ target_usd:
      risk_per_lot (profit ccy or account ccy) = (SL / tick_size) * tick_value
      -> convert to account ccy if needed, then lots = target_usd / risk_per_lot_in_account_ccy
    Enforces broker min/max/step.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    tick_size, tick_value, unit, profit_ccy = get_tick_specs(symbol)
    if not tick_size or not tick_value or tick_size == 0:
        return None

    ticks = sl_distance / tick_size
    risk_per_lot = ticks * float(tick_value)  # for 1.0 lot

    if unit == "profit":
        risk_per_lot_acct = convert_to_account_currency(risk_per_lot, profit_ccy, account_ccy)
    else:
        risk_per_lot_acct = risk_per_lot

    if not risk_per_lot_acct or risk_per_lot_acct <= 0:
        return None

    raw_lots = float(target_usd) / risk_per_lot_acct

    # Quantize
    step = info.volume_step or 0.01
    raw_lots = round(raw_lots / step) * step
    raw_lots = max(raw_lots, info.volume_min)
    raw_lots = min(raw_lots, info.volume_max)

    step_str = f"{step:.10f}".rstrip("0")
    dec = len(step_str.split(".")[1]) if "." in step_str else 0
    return float(f"{raw_lots:.{dec}f}")

def estimate_risk(symbol, sl_distance, lots, account_ccy="USD"):
    """
    Estimate risk (account_ccy and profit_ccy) for transparency.
    Returns (risk_in_account_ccy, profit_ccy, risk_in_profit_ccy) or (None, None, None).
    """
    info = mt5.symbol_info(symbol)
    if info is None:
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
def fetch_oanda_candles(symbol, granularity="H1", count=500):
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
        "price": "M"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"[ERROR] OANDA request exception for {symbol}: {e}")
        return None

    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch candles from OANDA {symbol}: {response.status_code} {response.text}")
        return None

    raw_candles = response.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}

    for candle in raw_candles:
        if candle.get('complete'):
            utc_time = pd.to_datetime(candle['time'])
            # Normalize to timezone-aware then convert to local_tz
            if utc_time.tzinfo is None:
                utc_time = utc_time.tz_localize('UTC')
            else:
                utc_time = utc_time.tz_convert('UTC')
            local_time = utc_time.tz_convert(local_tz)

            data['time'].append(local_time)
            data['open'].append(float(candle['mid']['o']))
            data['high'].append(float(candle['mid']['h']))
            data['low'].append(float(candle['mid']['l']))
            data['close'].append(float(candle['mid']['c']))
            data['volume'].append(int(candle['volume']))

    if not data['time']:
        return None

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df

# =================================================================================================
# ====== Trading helpers (MT5)
# =================================================================================================
def get_position(symbol, magic=None):
    """
    Retrieve existing position on a symbol. If 'magic' is provided and the broker supports hedging,
    only return the position with that magic. On netting accounts, returns the (sole) net position.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    if magic is None:
        return positions[0]
    for p in positions:
        if getattr(p, "magic", None) == magic:
            return p
    # IMPORTANT: don't touch other lanes' positions
    return None

def send_order(symbol, action_type, lot, sl_distance, tp_distance=None, magic=None, comment="ChandelierEntryBot"):
    """
    Submit a market order with fixed distance SL/TP off the current quote.
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

    # Quantize lot to broker steps
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
    sl_price = price - sl_distance if action_type == mt5.ORDER_TYPE_BUY else price + sl_distance
    tp_price = None if tp_distance is None else (price + tp_distance if action_type == mt5.ORDER_TYPE_BUY else price - tp_distance)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": price,
        "sl": sl_price,
        "deviation": slippage,
        "magic": magic if magic is not None else 987654,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if tp_price is not None:
        request["tp"] = tp_price

    result = mt5.order_send(request)
    if result is None:
        print(f"[ERROR] No response from order_send for {symbol}")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order failed on {symbol}: Retcode={result.retcode}, Comment={result.comment}")
    else:
        print(f"[OK] Order placed on {symbol}: Ticket={result.order} | Comment={comment}")

def close_position(position, symbol):
    """
    Close an open position (market close).
    """
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
# ====== Indicators: Heikin-Ashi + Chandelier-style directional stops
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
    """
    Produces:
      - ha_open/ha_high/ha_low/ha_close
      - EWM ATR(atr_period) on HA true ranges
      - smoothed chandelier stops and direction
      - boolean buy/sell flip signals on direction change
    """
    ha_df = calculate_heikin_ashi(df)

    tr = pd.DataFrame(index=ha_df.index)
    tr['ha_h'] = ha_df['ha_high']
    tr['ha_l'] = ha_df['ha_low']
    tr['ha_c'] = ha_df['ha_close']
    tr['prev_ha_c'] = tr['ha_c'].shift(1)

    # Include for debugging/visualisation
    tr['ha_open'] = ha_df['ha_open']
    tr['ha_high'] = ha_df['ha_high']
    tr['ha_low']  = ha_df['ha_low']

    # True Range & ATR (EWM)
    tr['true_range'] = tr[['ha_h', 'ha_l', 'prev_ha_c']].apply(
        lambda row: max(
            row['ha_h'] - row['ha_l'],
            abs(row['ha_h'] - row['prev_ha_c']) if pd.notna(row['prev_ha_c']) else 0.0,
            abs(row['ha_l'] - row['prev_ha_c']) if pd.notna(row['prev_ha_c']) else 0.0
        ), axis=1
    )
    tr['atr'] = tr['true_range'].ewm(alpha=1/atr_period, adjust=False).mean()

    # Chandelier Stops
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

    # Direction & signals
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

# =================================================================================================
# ====== Startup: show symbol info and risk examples (once)
# =================================================================================================
# We only print once per unique MT5 symbol to avoid spam.
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

# Optional: sample risk lines per strategy (comment out if too verbose)
for s in strategies:
    sym = s['mt5']
    lot = s.get('lot_size') or lot_size_default
    if s.get('risk_usd') is not None:
        dyn = lots_for_target_usd(sym, s['fixed_sl'], s['risk_usd'], account.currency)
        lot = dyn if dyn is not None else lot  # keep fallback if dyn fails

    r_acct, p_ccy, r_profit = estimate_risk(sym, s['fixed_sl'], lot, account.currency)
    if r_acct is not None:
        extra = f" (≈ {r_profit:.2f} {p_ccy})" if r_profit is not None else ""
        print(f"[RISK] {s['id']}: lot={lot} | SL={s['fixed_sl']} | TP={s['fixed_tp']} -> est ≈ {r_acct:.2f} {account.currency}{extra}")
    else:
        print(f"[RISK] {s['id']}: lot={lot} | SL={s['fixed_sl']} | TP={s['fixed_tp']} -> est risk unavailable")

# =================================================================================================
# ====== Per-strategy state (independent per (symbol, timeframe))
# =================================================================================================
last_candle_time = {s['id']: None for s in strategies}
last_signal      = {s['id']: None for s in strategies}
last_signal_info = {s['id']: None for s in strategies}

# =================================================================================================
# ====== MAIN HEARTBEAT LOOP
# - Checks each strategy independently
# - Acts only on *new* completed candles per timeframe
# - Hedging-style: no alignment between TFs, so they can disagree
# - Logs include the strategy id (e.g., [OK] XAU-M15 ...) for clarity by timeframe
# =================================================================================================
print("\n[Engine] Multi-timeframe mode: watching for new H1 and M15 candles...\n")
while True:
    for s in strategies:
        sid   = s['id']          # e.g., "XAU-H1" or "BTC-M15"
        o_sym = s['oanda']       # OANDA instrument
        m_sym = s['mt5']         # MT5 symbol
        tf    = s['granularity'] # "H1" or "M15"

        df = fetch_oanda_candles(symbol=o_sym, granularity=tf, count=num_candles)
        if df is None or df.empty:
            print(f"[ERROR] {sid}: no data")
            continue

        latest_candle_time = df.index[-1]
        if last_candle_time[sid] is not None and latest_candle_time <= last_candle_time[sid]:
            # No new candle for this TF yet
            continue
        last_candle_time[sid] = latest_candle_time
        print(f"[OK] {sid}: new {tf} candle @ {latest_candle_time.strftime('%Y-%m-%d %H:%M')}")

        # Calculate indicators FIRST (so we can log HA tables)
        tr = calculate_indicators(df)
        latest = tr.iloc[-1]

        # ===== Detailed logging (optional) =====
        if VERBOSE_LOG:
            # Raw OANDA OHLCV (last N)
            raw_tail = df[['open', 'high', 'low', 'close', 'volume']].copy().tail(PRINT_LAST_N)
            raw_tail.index = raw_tail.index.strftime('%Y-%m-%d %H:%M')
            print(f"\n===== RAW OANDA CANDLESTICK ({sid}, {tf}) — last {PRINT_LAST_N} =====")
            print(raw_tail)

            # Heikin-Ashi + direction/signal (last N)
            dbg = tr[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'buy_signal', 'sell_signal']].copy()
            dbg['signal'] = dbg.apply(lambda r: 'BUY' if r['buy_signal'] else ('SELL' if r['sell_signal'] else ''), axis=1)
            dbg.index = dbg.index.strftime('%Y-%m-%d %H:%M')
            print(f"\n===== HEIKIN-ASHI + SIGNALS ({sid}, {tf}) — last {PRINT_LAST_N} =====")
            print(dbg[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'signal']].tail(PRINT_LAST_N))
            print()  # spacer

        # Decide signal after logging
        signal = 'BUY' if latest['buy_signal'] else ('SELL' if latest['sell_signal'] else None)

        # Fetch existing position for this strategy (by magic for hedging accounts)
        position = get_position(m_sym, magic=s['magic'])
        open_pos = None
        if position:
            open_pos = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

        if not signal:
            print(f"[INFO] {sid}: no actionable signal.")
            continue

        # Trade only on fresh flip or if actual open position differs from desired side
        if signal != last_signal[sid] or open_pos != signal:
            prev_sig = last_signal[sid] if last_signal[sid] else "NONE"
            print(f"[TRADE] {sid}: new signal={signal} | prev={prev_sig} | open={open_pos or 'NONE'}")

            last_signal[sid] = signal
            last_signal_info[sid] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {signal}"

            # If flipping from opposite side, close that leg first (keeps clean per-strategy tickets on hedging accounts)
            if signal == 'BUY':
                if open_pos == 'SELL':
                    close_position(position, m_sym)
                if open_pos != 'BUY':
                    use_lot = s.get('lot_size') or lot_size_default
                    if s.get('risk_usd') is not None:
                        dyn = lots_for_target_usd(m_sym, s['fixed_sl'], s['risk_usd'], account.currency)
                        use_lot = dyn if dyn is not None else use_lot

                    send_order(m_sym, mt5.ORDER_TYPE_BUY, use_lot, s['fixed_sl'], s['fixed_tp'],
                               magic=s['magic'], comment=f"CEB:{sid}")

            elif signal == 'SELL':
                if open_pos == 'BUY':
                    close_position(position, m_sym)
                if open_pos != 'SELL':
                    use_lot = s.get('lot_size') or lot_size_default
                    if s.get('risk_usd') is not None:
                        dyn = lots_for_target_usd(m_sym, s['fixed_sl'], s['risk_usd'], account.currency)
                        use_lot = dyn if dyn is not None else use_lot

                    send_order(m_sym, mt5.ORDER_TYPE_SELL, use_lot, s['fixed_sl'], s['fixed_tp'],
                               magic=s['magic'], comment=f"CEB:{sid}")
        else:
            print(f"[INFO] {sid}: signal unchanged ({signal}).")

    # Throttle API calls. M15 candles close every 15 minutes; 20s polling is more than enough.
    time.sleep(20)
