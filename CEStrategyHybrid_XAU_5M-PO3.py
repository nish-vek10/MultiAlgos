"""
XAUUSD | M5 | Heikin-Ashi + Chandelier (NORMAL+REVERSE auto-switch via PO3-style regime)
- Always-on (sessions optional; default = None => 24/5)
- Risk = 0.50% of CURRENT ACCOUNT BALANCE per trade
- NORMAL (trend):   SL=5.5,  TP1=6.5,  TP1=60%, then SL -> BE + buffer (>= spread)
- REVERSE (mean-rv): SL=3.0,  TP1=3.5,  TP1=60%, then SL -> BE + buffer (>= spread)
- Pending/retry, backfill prev bar, post-fill SL adjust to filled price
- Clean logs; easy to tweak constants below

Fill in your MT5 and OANDA credentials in the CONFIG section.
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone, UTC

import requests
import MetaTrader5 as mt5

# =========================
# ===== USER CONFIG =======
# =========================

# --- Symbols / TF ---
MT5_SYMBOL        = "XAUUSD"
OANDA_SYMBOL      = "XAU_USD"
OANDA_GRANULARITY = "M5"
NUM_CANDLES       = 500

# --- Local/logging TZ ---
LOCAL_TZ = timezone('Europe/London')

# --- MT5 login (EDIT THESE) ---
MT5_LOGIN         = 52561010
MT5_PASSWORD      = "k!9PSuIdfEiP$3"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\EA-XAU-PO3\terminal64.exe"

# --- OANDA API (EDIT THESE) ---
OANDA_TOKEN       = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
OANDA_API_BASE    = "https://api-fxpractice.oanda.com/v3"

# --- Trade engine ---
SLIPPAGE               = 10
MAGIC_NUMBER           = 556677
RETRY_EVERY_SECS       = 15
USE_HEIKIN_ASHI        = True

# --- Risk (CURRENT BALANCE) ---
RISK_PCT_OF_BALANCE    = 0.50   # per trade, % of current balance

# --- NORMAL (trend) regime params ---
NORMAL_SL_USD          = 5.5
NORMAL_TP1_USD         = 6.5
NORMAL_TP1_FRACTION    = 0.60
BE_BUFFER_USD          = 0.20   # base buffer for BE; actual = max(BE_BUFFER_USD, spread)

# --- REVERSE (mean-revert) regime params ---
REVERSE_SL_USD         = 3.0
REVERSE_TP1_USD        = 3.5
REVERSE_TP1_FRACTION   = 0.60

# --- Chandelier / ATR (same for both regimes; HA is applied upstream if USE_HEIKIN_ASHI=True) ---
ATR_PERIOD             = 1
ATR_MULT               = 1.85

# --- Sessions (optional). Set to None for always-on 24/5. ---
SESSION_WINDOWS = None
AUTO_CLOSE_AT_SESSION_END = False
REQUIRE_POST_START_CANDLE = False

# --- Regime detection (PO3-style, tweak if needed) ---
# "PO3" idea implemented as: trendiness score (dir consistency + EMA slope + displacement)
# AND manipulation/sweep checks (recent sweep of prior high/low and close back inside).
# If trendiness strong and no fresh sweep → NORMAL (trend). Else → REVERSE (mean-revert).

# === PO3 constants to match the backtest exactly ===
PERSISTENCE_K   = 12      # run-length of unbroken CE direction
SMA_LEN         = 20
ATR_TREND_LEN   = 20
DIST_ATR_MULT   = 1.00
SWEEP_LOOKBACK  = 288     # ~24h on M5


# =========================
# ====== LOG HELPER =======
# =========================
def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} | {msg}", flush=True)

# =========================
# == MT5 BOOTSTRAPPING ====
# =========================
print("MT5 Path Exists?", os.path.exists(MT5_TERMINAL_PATH))
if not mt5.initialize(login=int(MT5_LOGIN) if str(MT5_LOGIN).isdigit() else None,
                      password=MT5_PASSWORD,
                      server=MT5_SERVER,
                      path=MT5_TERMINAL_PATH):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    sys.exit(1)

acct = mt5.account_info()
if acct is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")
print(f"\nMT5 ACCOUNT CONNECTED!\nACCOUNT: {acct.login}\nBALANCE: ${acct.balance:.2f}\n")

si = mt5.symbol_info(MT5_SYMBOL)
if si:
    print(f"[INFO] {MT5_SYMBOL} Lot Range: min={si.volume_min}, max={si.volume_max}, step={si.volume_step}")
    min_sl_usd = (getattr(si, "trade_stops_level", 0) or 0) * si.point
    print(f"[INFO] Approx. min stop distance enforced by broker ≈ ${min_sl_usd:.2f}")
else:
    print(f"[ERROR] Unable to fetch symbol info for {MT5_SYMBOL}")


# =========================
# ===== OANDA FETCH =======
# =========================
def fetch_oanda_candles(symbol=OANDA_SYMBOL, granularity=OANDA_GRANULARITY, count=NUM_CANDLES):
    """
    Fetch candle data from OANDA REST API using requests.get.
    Returns a pandas DataFrame indexed by LOCAL_TZ time (only complete candles).
    """
    url = f"{OANDA_API_BASE}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"granularity": granularity, "count": count, "price": "M"}  # Mid prices

    try:
        r = requests.get(url, headers=headers, params=params, timeout=(5, 15))
    except requests.RequestException as e:
        print(f"[ERROR] OANDA network error: {e.__class__.__name__}: {e}")
        return None

    if r.status_code != 200:
        print("[ERROR] Failed to fetch candles from OANDA:", r.status_code, r.text[:300])
        return None

    raw = r.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for c in raw:
        if c.get("complete", False):
            utc_time = pd.to_datetime(c["time"], utc=True)
            local_time = utc_time.tz_convert(LOCAL_TZ)
            data["time"].append(local_time)
            data["open"].append(float(c["mid"]["o"]))
            data["high"].append(float(c["mid"]["h"]))
            data["low"].append(float(c["mid"]["l"]))
            data["close"].append(float(c["mid"]["c"]))
            data["volume"].append(int(c["volume"]))

    if not data["time"]:
        return None
    return pd.DataFrame(data).set_index("time")

# =========================
# ===== INDICATORS ========
# =========================
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

def calculate_indicators(df, useHeikinAshi=True, atrPeriod=1, atrMult=1.85):
    """
    Heikin-Ashi + True Range + RMA ATR + Chandelier-style stop smoothing + flips
    """
    if useHeikinAshi:
        ha = calculate_heikin_ashi(df)
        o = ha['ha_open']; h = ha['ha_high']; l = ha['ha_low']; c = ha['ha_close']
    else:
        o = df['open']; h = df['high']; l = df['low']; c = df['close']

    tr = pd.DataFrame(index=df.index)
    tr['o'] = o; tr['h'] = h; tr['l'] = l; tr['c'] = c
    tr['c_prev'] = tr['c'].shift(1)

    # true range like Pine
    def _tr(row):
        if pd.isna(row['c_prev']): return row['h'] - row['l']
        return max(row['h'] - row['l'], abs(row['h'] - row['c_prev']), abs(row['l'] - row['c_prev']))
    tr['true_range'] = tr.apply(_tr, axis=1)

    # RMA ATR with SMA seed
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

    # raw CE stops
    hh = tr['h'].rolling(window=n, min_periods=n).max()
    ll = tr['l'].rolling(window=n, min_periods=n).min()
    long_stop  = hh - atr_val
    short_stop = ll + atr_val

    # smooth stops (sticky)
    lss = long_stop.copy()
    sss = short_stop.copy()
    for i in range(len(tr)):
        if i == 0: continue
        long_prev  = lss.iloc[i-1] if pd.notna(lss.iloc[i-1]) else long_stop.iloc[i]
        short_prev = sss.iloc[i-1] if pd.notna(sss.iloc[i-1]) else short_stop.iloc[i]
        if tr['c'].iloc[i-1] > long_prev:   lss.iloc[i] = max(long_stop.iloc[i], long_prev)
        else:                               lss.iloc[i] = long_stop.iloc[i]
        if tr['c'].iloc[i-1] < short_prev:  sss.iloc[i] = min(short_stop.iloc[i], short_prev)
        else:                               sss.iloc[i] = short_stop.iloc[i]

    # direction & flips using previous smoothed stops (dir is sticky)
    dir_vals = [1]
    for i in range(1, len(tr)):
        if tr['c'].iloc[i] > sss.iloc[i-1]: dir_vals.append(1)
        elif tr['c'].iloc[i] < lss.iloc[i-1]: dir_vals.append(-1)
        else: dir_vals.append(dir_vals[-1])

    tr['dir'] = dir_vals
    tr['dir_prev'] = tr['dir'].shift(1)
    tr['buy_signal']  = (tr['dir'] ==  1) & (tr['dir_prev'] == -1)
    tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] ==  1)

    # expose for debug
    tr['long_stop_smooth']  = lss
    tr['short_stop_smooth'] = sss
    tr['ha_open'] = o; tr['ha_high'] = h; tr['ha_low'] = l; tr['ha_c'] = c

    return tr


# ======= Backtest-accurate helpers =======

def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA with SMA seed — same as the backtest."""
    if length <= 1:
        return series.copy()
    alpha = 1.0 / length
    out = series.copy().astype(float)
    seed = series.rolling(length, min_periods=length).mean()
    out.iloc[:length-1] = float('nan')
    out.iloc[length-1] = seed.iloc[length-1]
    for i in range(length, len(series)):
        out.iloc[i] = out.iloc[i-1] + alpha * (series.iloc[i] - out.iloc[i-1])
    return out.ffill()

def _true_range_from_ohlc(df: pd.DataFrame) -> pd.Series:
    c = df['close']
    h = df['high']; l = df['low']; c_prev = c.shift(1)
    tr1 = (h - l)
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _sweep_recent_by_prev_day_hilo(df: pd.DataFrame, lookback: int) -> bool:
    """
    Backtest-like 'sweep' of PRIOR-DAY high/low.
    We compute prev-day high/low per bar, then flag if any of last `lookback` bars
    took out that prev-day level.
    """
    if len(df) < max(3, lookback + 2):
        return False
    # group by UTC day (stable across DST)
    day = df.index.tz_convert("UTC").floor("D")
    daily_high_prev = df["high"].groupby(day).transform("max").shift(1)
    daily_low_prev  = df["low"].groupby(day).transform("min").shift(1)
    swept = (df["high"] > daily_high_prev) | (df["low"] < daily_low_prev)
    # any sweep in recent window?
    return bool(swept.tail(lookback).max())


# =========================
# ==== PO3 REGIME DET =====
# =========================
def detect_regime(df_raw: pd.DataFrame, ce_df: pd.DataFrame):
    """
    EXACT match to backtest heuristic:
      - Persistence (run-length) of CE dir >= PERSISTENCE_K
      - Displacement: |close - SMA20| >= ATR20 * DIST_ATR_MULT
      - Sweep nudge: if recent sweep of prior-day H/L within SWEEP_LOOKBACK -> NORMAL
    Returns: ("NORMAL" | "REVERSE", diagnostics_dict)
    """
    diag = {}
    try:
        if df_raw is None or df_raw.empty or ce_df is None or ce_df.empty:
            return "NORMAL", {"error": "empty df"}

        # --- Persistence via run-length on ce_df['dir'] ---
        dir_series = ce_df['dir'].fillna(0).astype(int)
        # build run lengths
        switches = (dir_series != dir_series.shift(1)).cumsum()
        run_lengths = switches.groupby(switches).transform('count')
        persistence_ok = bool(run_lengths.iloc[-1] >= PERSISTENCE_K)
        diag['persistence_run'] = int(run_lengths.iloc[-1])

        # --- Displacement vs ATR20 using SMA20 and RMA(TR,20) ---
        c = df_raw['close']
        sma20 = c.rolling(SMA_LEN, min_periods=1).mean()
        tr = _true_range_from_ohlc(df_raw)
        atr20 = _rma(tr, ATR_TREND_LEN)
        last_disp = abs(c.iloc[-1] - sma20.iloc[-1])
        last_atr  = float(atr20.iloc[-1]) if pd.notna(atr20.iloc[-1]) else 0.0
        disp_ok = (last_atr > 0.0) and (last_disp >= last_atr * DIST_ATR_MULT)

        diag['disp_abs'] = float(last_disp)
        diag['atr20']    = float(last_atr)
        diag['disp_ok']  = bool(disp_ok)

        # --- Sweep nudge (if recent sweep -> NORMAL) ---
        swept_recent = _sweep_recent_by_prev_day_hilo(df_raw, SWEEP_LOOKBACK)
        diag['sweep_recent'] = bool(swept_recent)

        trend_ok = persistence_ok and disp_ok
        regime = "NORMAL" if (trend_ok or swept_recent) else "REVERSE"

        diag['trend_ok'] = bool(trend_ok)
        diag['decision'] = regime
        return regime, diag

    except Exception as e:
        diag['error'] = str(e)
        # neutral default like backtest (leans NORMAL)
        return "NORMAL", diag


# =========================
# ==== SESSION HELPERS ====
# =========================
def _make_dt(base_dt, hhmm, tz):
    h, m = map(int, hhmm.split(":"))
    return base_dt.astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0)

def in_session(now_local):
    if SESSION_WINDOWS is None:
        return True
    for start, end in SESSION_WINDOWS:
        start_dt = _make_dt(now_local, start, LOCAL_TZ)
        end_dt   = _make_dt(now_local, end,   LOCAL_TZ)
        if start_dt <= now_local < end_dt:
            return True
    return False

def current_session_bounds(now_local):
    if SESSION_WINDOWS is None:
        return (None, None)
    for start, end in SESSION_WINDOWS:
        start_dt = _make_dt(now_local, start, LOCAL_TZ)
        end_dt   = _make_dt(now_local, end,   LOCAL_TZ)
        if start_dt <= now_local < end_dt:
            return start_dt, end_dt
    return (None, None)

def _next_5m_close(now):
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    target = base + timedelta(minutes=mins_to_add)
    if target <= now:
        target += timedelta(minutes=5)
    return target

# =========================
# ===== RISK / SIZING =====
# =========================
def _round_to_step(value, step):
    steps = math.floor(value / step)
    return max(steps * step, 0.0)

def _value_per_1usd_per_lot(si):
    if getattr(si, "trade_tick_value", 0) and getattr(si, "trade_tick_size", 0):
        return float(si.trade_tick_value) / float(si.trade_tick_size)
    if getattr(si, "tick_value", 0) and getattr(si, "tick_size", 0):
        return float(si.tick_value) / float(si.tick_size)
    if getattr(si, "tick_value", 0) and getattr(si, "point", 0):
        return float(si.tick_value) / float(si.point)
    if getattr(si, "trade_contract_size", 0):
        return float(si.trade_contract_size)
    return None

# Sanity: $ per $1 move at 1.0 lot
try:
    v = _value_per_1usd_per_lot(si)
    if v:
        print(f"[CHECK] $ per $1 move (1 lot) detected: {v:.2f}")
        if not (98.0 <= v <= 102.0):
            print("[WARN] Detected contract value differs from backtest's $100. "
                  "Forward position sizes will not match backtest exactly.")
except Exception as _e:
    print(f"[WARN] Could not compute $/lot check: {_e}")

def compute_lot_for_risk_dynamic_balance(symbol, sl_usd, risk_pct_of_balance: float):
    si = mt5.symbol_info(symbol)
    acc = mt5.account_info()
    if si is None or acc is None:
        print("[ERROR] Sizing: symbol or account info not available.")
        return None, 0.0

    balance = float(acc.balance)
    risk_amount = balance * (float(risk_pct_of_balance) / 100.0)
    if risk_amount <= 0:
        return None, 0.0

    v = _value_per_1usd_per_lot(si)
    if not v:
        print("[ERROR] Cannot derive $ value per $1 move for 1 lot; missing tick fields.")
        return None, 0.0

    risk_per_lot = v * float(sl_usd)
    if risk_per_lot <= 0:
        return None, 0.0

    raw_lot = risk_amount / risk_per_lot
    lot = _round_to_step(raw_lot, si.volume_step)
    lot = max(si.volume_min, min(lot, si.volume_max))
    return lot, risk_amount

def compute_sl_price(symbol, action_type, entry_price, sl_usd):
    si = mt5.symbol_info(symbol)
    if si is None: return None
    digits = si.digits
    point  = si.point
    if action_type == mt5.ORDER_TYPE_BUY:
        sl = entry_price - float(sl_usd)
    else:
        sl = entry_price + float(sl_usd)
    # respect broker min distance
    stops_pts = getattr(si, "trade_stops_level", 0)
    if stops_pts and stops_pts > 0:
        min_dist = stops_pts * point
        if abs(entry_price - sl) < min_dist:
            sl = entry_price - min_dist if action_type == mt5.ORDER_TYPE_BUY else entry_price + min_dist
    return round(sl, digits)

def compute_tp_price(symbol, action_type, entry_price, tp_usd):
    si = mt5.symbol_info(symbol)
    if si is None: return None
    digits = si.digits
    if action_type == mt5.ORDER_TYPE_BUY:
        tp = entry_price + float(tp_usd)
    else:
        tp = entry_price - float(tp_usd)
    return round(tp, digits)

# =========================
# ==== POS MANAGEMENT =====
# =========================
def _get_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if int(getattr(p, "magic", 0)) == int(MAGIC_NUMBER):
            return p
    return None

def _round_volume_to_step(si, vol):
    vol = math.floor(vol / si.volume_step) * si.volume_step
    return max(si.volume_min, min(vol, si.volume_max))

def _partial_close(position, fraction):
    """Close a fraction of the position at market; return True on success."""
    si = mt5.symbol_info(position.symbol)
    if si is None:
        return False

    part_vol_raw = position.volume * float(fraction)
    part_vol = _round_volume_to_step(si, part_vol_raw)

    remain_vol = round(position.volume - part_vol, 8)
    if remain_vol > 0 and remain_vol < si.volume_min:
        # ensure runner >= min
        min_keep = si.volume_min
        part_vol = _round_volume_to_step(si, position.volume - min_keep)
        remain_vol = round(position.volume - part_vol, 8)

    if part_vol <= 0:
        return False

    opp_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return False
    price = tick.bid if opp_type == mt5.ORDER_TYPE_SELL else tick.ask

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": part_vol,
        "type": opp_type,
        "position": position.ticket,
        "price": price,
        "deviation": SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "PartialTP60",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[OK] Partial closed {part_vol} from ticket {position.ticket}")
        return True
    print(f"[WARN] Partial close failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    return False

def _move_sl_to_breakeven(position, buffer_usd=0.0):
    """
    Move SL to entry ± buffer if broker min distance allows it. Returns True if moved.
    """
    symbol = position.symbol
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        return False

    entry = float(position.price_open)
    point = si.point
    stops_pts = getattr(si, "trade_stops_level", 0) or 0
    min_dist = stops_pts * point

    be_buf = float(buffer_usd)

    if position.type == mt5.ORDER_TYPE_BUY:
        desired_sl = round(entry + be_buf, si.digits)
        max_allowed_sl = tick.bid - min_dist if min_dist > 0 else tick.bid
        if desired_sl <= max_allowed_sl and desired_sl < tick.bid:
            mod = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": position.ticket,
                "sl": desired_sl,
                "magic": MAGIC_NUMBER,
                "comment": "SL->BE",
            }
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] SL moved to BE+buffer at {desired_sl} (buy).")
                return True
            print(f"[WARN] BE SL set failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False
    else:
        desired_sl = round(entry - be_buf, si.digits)
        min_allowed_sl = tick.ask + min_dist if min_dist > 0 else tick.ask
        if desired_sl >= min_allowed_sl and desired_sl > tick.ask:
            mod = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": position.ticket,
                "sl": desired_sl,
                "magic": MAGIC_NUMBER,
                "comment": "SL->BE",
            }
            res = mt5.order_send(mod)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OK] SL moved to BE+buffer at {desired_sl} (sell).")
                return True
            print(f"[WARN] BE SL set failed: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
            return False
        return False

def _close_position(position):
    """Close an open position market."""
    if position is None:
        return
    opp_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return
    price = tick.bid if opp_type == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": opp_type,
        "position": position.ticket,
        "price": price,
        "deviation": SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "Close opposite position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {getattr(res,'retcode',None)} {getattr(res,'comment',None)}")
    else:
        print(f"[OK] POSITION CLOSED: {res}")

# =========================
# ===== ORDER SENDER ======
# =========================
def send_order(symbol, action_type, sl_usd, risk_pct):
    """
    Market order, risk-sized by current balance, post-fill SL (to filled-price distance).
    """
    # ensure symbol is tradable
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol {symbol}")
        return False

    si = mt5.symbol_info(symbol)
    if si is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return False

    if hasattr(si, "trade_allowed") and not si.trade_allowed:
        print(f"[ERROR] Trading is not allowed for {symbol}")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return False

    entry_price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    # snapshot SL distance for sizing
    sl_price_snapshot = compute_sl_price(symbol, action_type, entry_price, sl_usd)
    if sl_price_snapshot is None:
        print("[ERROR] Could not compute SL price.")
        return False
    sl_distance = abs(entry_price - sl_price_snapshot)

    # lot sizing
    lot, risk_amount = compute_lot_for_risk_dynamic_balance(symbol, sl_distance, risk_pct)
    if lot is None or lot <= 0:
        print("[ERROR] Computed lot is invalid; aborting order.")
        return False
    print(f"[RISK] Risk: {risk_pct:.2f}% of BAL = ${risk_amount:,.2f} | SL dist: ${sl_distance:.2f} | Lot: {lot}")

    # try common fill modes
    modes = []
    if hasattr(mt5, "ORDER_FILLING_RETURN"): modes.append(mt5.ORDER_FILLING_RETURN)
    if hasattr(mt5, "ORDER_FILLING_FOK"):    modes.append(mt5.ORDER_FILLING_FOK)
    if hasattr(mt5, "ORDER_FILLING_IOC"):    modes.append(mt5.ORDER_FILLING_IOC)
    if not modes: modes = [mt5.ORDER_FILLING_IOC]

    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": entry_price,
        "deviation": SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "CE-PO3",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    last_res = None
    for fm in modes:
        req = dict(req_base, type_filling=fm)
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] ORDER PLACED (fill_mode={fm}): ticket={res.order}, price={entry_price}, lot={lot}")
            # Post-fill SL at filled price
            pos = _get_position(symbol)
            if pos:
                filled = pos.price_open
                desired_sl = compute_sl_price(symbol, action_type, filled, sl_usd)
                if desired_sl:
                    mod = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": pos.ticket,
                        "sl": desired_sl,
                        "magic": MAGIC_NUMBER,
                        "comment": "Set SL post-fill",
                    }
                    mod_res = mt5.order_send(mod)
                    if mod_res is not None and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"[OK] SL SET to {desired_sl}")
                    else:
                        print(f"[WARN] SL set failed: {getattr(mod_res,'retcode',None)} {getattr(mod_res,'comment',None)}")
            else:
                print("[WARN] Position not found after fill; cannot set SL.")
            return True
        last_res = res

    print(f"[ERROR] ORDER FAILED across all filling modes.")
    if last_res:
        print(f"        retcode={last_res.retcode}, comment={last_res.comment}")
    else:
        print(f"        order_send returned None. last_error={mt5.last_error()}")
    return False

# =========================
# == SIGNAL EXEC/RETRY ====
# =========================
def _attempt_execution_for_signal(desired_side: str, sl_usd: float):
    """
    Close opposite if needed; place market order risk-sized; returns True if correct side is realized or filled now.
    """
    position = _get_position(MT5_SYMBOL)
    open_side = None
    if position:
        open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

    if open_side == desired_side:
        return True

    if open_side and open_side != desired_side:
        _close_position(position)
        time.sleep(0.5)  # let server update

    order_type = mt5.ORDER_TYPE_BUY if desired_side == 'BUY' else mt5.ORDER_TYPE_SELL
    return bool(send_order(MT5_SYMBOL, order_type, sl_usd=sl_usd, risk_pct=RISK_PCT_OF_BALANCE))

def _price_side_touched(tick, target_price, pos_type):
    if pos_type == mt5.ORDER_TYPE_BUY:
        return tick.bid >= target_price
    return tick.ask <= target_price

# pending retry state
pending_signal = {"signal": None, "since": None, "last_retry": None, "sl_usd": None}

def _maybe_retry_pending():
    if not pending_signal["signal"]:
        return
    now_local = datetime.now(LOCAL_TZ)
    if not in_session(now_local):
        return
    if pending_signal["last_retry"] and (now_local - pending_signal["last_retry"]).total_seconds() < RETRY_EVERY_SECS:
        return

    print(f"[RETRY] Pending {pending_signal['signal']} — attempting execution…")

    # Use the SL distance captured when the signal was queued (parity with backtest)
    sl_cur = pending_signal.get("sl_usd", NORMAL_SL_USD)
    ok = _attempt_execution_for_signal(pending_signal["signal"], sl_usd=sl_cur)

    pending_signal["last_retry"] = now_local
    if ok:
        print(f"[OK] Pending {pending_signal['signal']} executed.")
        pending_signal.update({"signal": None, "since": None, "last_retry": None, "sl_usd": None})


# =========================
# ===== MAIN LOOPING ======
# =========================
last_candle_time = None
saw_candle_after_session_start = False

# per-position mgmt flags: ticket -> {"partial60_done": bool, "moved_to_be": bool, "regime_at_entry": "NORMAL"/"REVERSE"}
pos_state = {}

print("[BOOT] CE-PO3 M5 engine active. Always-on:", SESSION_WINDOWS is None)

try:
    while True:
        # 1) Stay in session (or always-on)
        while True:
            now_local = datetime.now(LOCAL_TZ)
            if in_session(now_local):
                sess_start, sess_end = current_session_bounds(now_local)
                saw_candle_after_session_start = False
                if sess_start and sess_end:
                    log(f"[+] IN SESSION: {sess_start.strftime('%H:%M:%S %Z')}–{sess_end.strftime('%H:%M:%S %Z')}")
                else:
                    log("[+] IN SESSION: always-on")
                break
            time.sleep(1)

        # 2) Wait to next 5-min close, retry pending + manage open while waiting
        target_close = _next_5m_close(datetime.now(LOCAL_TZ))
        if SESSION_WINDOWS is not None and sess_end and target_close > sess_end:
            target_close = sess_end
            log(f"[*] Session ends before next close → waiting until {target_close.strftime('%H:%M:%S %Z')} ...")
        else:
            log(f"[*] Waiting until next 5-min close at {target_close.strftime('%H:%M:%S %Z')} ...")

        while True:
            _maybe_retry_pending()
            # manage TP1/BE on the fly
            pos = _get_position(MT5_SYMBOL)
            if pos:
                # determine regime at entry if stored; else infer current regime for mgmt thresholds
                st = pos_state.get(pos.ticket, {"partial60_done": False, "moved_to_be": False, "regime_at_entry": None})
                regime_for_mgmt = st.get("regime_at_entry")
                if regime_for_mgmt is None:
                    # fallback to current regime
                    df_tmp = fetch_oanda_candles()
                    if df_tmp is not None and not df_tmp.empty:
                        tr_tmp = calculate_indicators(df_tmp, useHeikinAshi=USE_HEIKIN_ASHI, atrPeriod=ATR_PERIOD, atrMult=ATR_MULT)
                        regime_for_mgmt, _ = detect_regime(df_tmp[['open','high','low','close','volume']], tr_tmp)
                    else:
                        regime_for_mgmt = "NORMAL"
                tp1_d  = NORMAL_TP1_USD if regime_for_mgmt == "NORMAL" else REVERSE_TP1_USD
                tp1_fr = NORMAL_TP1_FRACTION if regime_for_mgmt == "NORMAL" else REVERSE_TP1_FRACTION

                tick = mt5.symbol_info_tick(MT5_SYMBOL)
                si   = mt5.symbol_info(MT5_SYMBOL)
                if tick and si:
                    # seed pos_state if not present
                    if pos.ticket not in pos_state:
                        pos_state[pos.ticket] = {"partial60_done": False, "moved_to_be": False, "regime_at_entry": regime_for_mgmt}
                        tp1_preview = compute_tp_price(pos.symbol, pos.type, pos.price_open, tp1_d)
                        print(f"[INFO] Managing ticket {pos.ticket}: entry={pos.price_open}, TP1(60%)={tp1_preview}, regime={regime_for_mgmt}")

                    # check TP1
                    tp1_price = compute_tp_price(pos.symbol, pos.type, pos.price_open, tp1_d)
                    if tp1_price is not None:
                        st = pos_state.get(pos.ticket)
                        if st and not st["partial60_done"]:
                            if _price_side_touched(tick, tp1_price, pos.type):
                                if _partial_close(pos, tp1_fr):
                                    st["partial60_done"] = True
                                    pos_state[pos.ticket] = st
                                    # refresh pos object
                                    time.sleep(0.30)
                                    pos = _get_position(MT5_SYMBOL)
                                    tick = mt5.symbol_info_tick(MT5_SYMBOL)
                    # BE move (retry until allowed)
                    st = pos_state.get(pos.ticket)
                    if st and st["partial60_done"] and not st["moved_to_be"]:
                        if _move_sl_to_breakeven(pos, BE_BUFFER_USD):
                            st["moved_to_be"] = True
                            pos_state[pos.ticket] = st

            now = datetime.now(LOCAL_TZ)
            if (target_close - now).total_seconds() <= 0:
                break
            time.sleep(0.25)
        time.sleep(2)  # allow API bar finalize

        # 3) Fetch fresh data and ensure truly new candle
        df = None
        retry_timeout = timedelta(seconds=20)
        start_poll = datetime.now(LOCAL_TZ)

        while True:
            df = fetch_oanda_candles()
            if df is None or df.empty:
                time.sleep(2)
                _maybe_retry_pending()
                if datetime.now(LOCAL_TZ) - start_poll > retry_timeout:
                    log("[TIMEOUT] No data from OANDA. Skipping cycle.")
                    df = None
                    break
                continue

            latest_ts = df.index[-1]
            if last_candle_time is not None and latest_ts <= last_candle_time:
                log("[WAIT] No new candle yet. Retrying in 2s…")
                time.sleep(2)
                _maybe_retry_pending()
                if datetime.now(LOCAL_TZ) - start_poll > retry_timeout:
                    log("[TIMEOUT] No new candle. Skipping cycle.")
                    df = None
                    break
                continue

            last_candle_time = latest_ts
            log(f"[OK] New 5-min candle: {latest_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            break

        if df is None or df.empty:
            continue

        # optional session first-post-candle guard
        if REQUIRE_POST_START_CANDLE and not saw_candle_after_session_start:
            start_dt, _ = current_session_bounds(datetime.now(LOCAL_TZ))
            if start_dt and last_candle_time <= start_dt:
                log("[WAIT] First post-session bar not closed yet; skipping...")
                continue
            saw_candle_after_session_start = True

        # 4) Print raw & compute indicators
        try:
            raw_to_show = df[['open','high','low','close','volume']].copy()
            raw_to_show.index = raw_to_show.index.strftime('%Y-%m-%d %H:%M')
            print("\n===== RAW OANDA M5 DATA — last 10 =====")
            print(raw_to_show.tail(10))
        except Exception as e:
            log(f"[DEBUG] Raw candle print failed: {e}")

        tr = calculate_indicators(df, useHeikinAshi=USE_HEIKIN_ASHI, atrPeriod=ATR_PERIOD, atrMult=ATR_MULT)
        latest = tr.iloc[-1]
        try:
            debug_df = tr[['ha_c','ha_open','ha_high','ha_low','dir','buy_signal','sell_signal']].copy()
            debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
            debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
            print("\n===== HA + SIGNALS (M5) — last 10 =====")
            print(debug_df[['ha_c','ha_open','ha_high','ha_low','dir','signal']].tail(10))
        except Exception as e:
            log(f"[DEBUG] HA table print failed: {e}")

        # 5) Regime detection (PO3-style)
        regime, diag = detect_regime(df[['open','high','low','close','volume']], tr)
        print(f"[REGIME] {regime} | diag: {diag}")

        # Map regime -> SL/TP1/fraction
        if regime == "NORMAL":
            SL_USD = NORMAL_SL_USD
            TP1_USD = NORMAL_TP1_USD
            TP1_FR  = NORMAL_TP1_FRACTION
        else:
            SL_USD = REVERSE_SL_USD
            TP1_USD = REVERSE_TP1_USD
            TP1_FR  = REVERSE_TP1_FRACTION

        # 6) Signals (raw), then regime mapping (REVERSE flips signals)
        raw_signal = 'BUY' if bool(latest['buy_signal']) else ('SELL' if bool(latest['sell_signal']) else None)
        if regime == "NORMAL":
            signal = raw_signal
        else:
            # invert
            if raw_signal == 'BUY':
                signal = 'SELL'
            elif raw_signal == 'SELL':
                signal = 'BUY'
            else:
                signal = None

        # previous-bar for backfill (with regime mapping applied consistently)
        prev_signal = None
        if len(tr) >= 2:
            prev = tr.iloc[-2]
            raw_prev = 'BUY' if prev['buy_signal'] else ('SELL' if prev['sell_signal'] else None)
            if regime == "NORMAL":
                prev_signal = raw_prev
            else:
                if raw_prev == 'BUY': prev_signal = 'SELL'
                elif raw_prev == 'SELL': prev_signal = 'BUY'
                else: prev_signal = None

        # position info
        position = _get_position(MT5_SYMBOL)
        open_side = None
        if position:
            open_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
            print(f"[INFO] OPEN POSITION: {open_side}, vol: {position.volume}, entry: {position.price_open}")
            # seed pos_state if new
            if position.ticket not in pos_state:
                tp1_preview = compute_tp_price(position.symbol, position.type, position.price_open, TP1_USD)
                pos_state[position.ticket] = {"partial60_done": False, "moved_to_be": False, "regime_at_entry": regime}
                print(f"[INFO] Manage ticket {position.ticket}: entry={position.price_open}, TP1(60%)={tp1_preview}, regime={regime}")
        else:
            print("[INFO] No open position currently.")

        # 7) Execute with pending/backfill/override (and store regime at entry)
        now_local = datetime.now(LOCAL_TZ)

        if signal:
            if pending_signal["signal"] and signal != pending_signal["signal"]:
                print(f"[CANCELLED] Live signal {signal} overrides pending {pending_signal['signal']}.")
                ok = _attempt_execution_for_signal(signal, sl_usd=SL_USD)
                if ok:
                    pending_signal.update({"signal": None, "since": None, "last_retry": None})
                    # attach regime flag to the (new) position
                    pos = _get_position(MT5_SYMBOL)
                    if pos:
                        pos_state[pos.ticket] = {"partial60_done": False, "moved_to_be": False, "regime_at_entry": regime}
                else:
                    pending_signal.update({"signal": signal, "since": now_local if pending_signal["since"] is None else pending_signal["since"]})
            else:
                if open_side != signal:
                    print(f"[TRADE] New signal={signal} | open={open_side or 'NONE'} | regime={regime}")
                    ok = _attempt_execution_for_signal(signal, sl_usd=SL_USD)
                    if ok:
                        pending_signal.update({"signal": None, "since": None, "last_retry": None})
                        pos = _get_position(MT5_SYMBOL)
                        if pos:
                            pos_state[pos.ticket] = {"partial60_done": False, "moved_to_be": False, "regime_at_entry": regime}
                    else:
                        if not pending_signal["signal"]:
                            pending_signal.update({"signal": signal, "since": now_local, "sl_usd": SL_USD})
                            print(f"[PENDING] Queued {signal} (will retry).")
        else:
            # consider backfill prev bar
            if prev_signal and open_side != prev_signal and not pending_signal["signal"]:
                print(f"[BACKFILL] Previous bar had {prev_signal} — attempting execution now (regime={regime}).")
                ok = _attempt_execution_for_signal(prev_signal, sl_usd=SL_USD)
                if ok:
                    pending_signal.update({"signal": None, "since": None, "last_retry": None})
                    pos = _get_position(MT5_SYMBOL)
                    if pos:
                        pos_state[pos.ticket] = {"partial60_done": False, "moved_to_be": False, "regime_at_entry": regime}
                else:
                    pending_signal.update({"signal": prev_signal, "since": now_local, "sl_usd": SL_USD})
                    print(f"[PENDING] Queued {prev_signal} from previous bar (will retry).")
            else:
                print("[INFO] No actionable signal.")

        # 8) Session end handling (optional)
        if SESSION_WINDOWS is not None:
            now_local = datetime.now(LOCAL_TZ)
            _, sess_end = current_session_bounds(now_local)
            if sess_end and now_local >= sess_end:
                log("[INFO] SESSION JUST ENDED.")
                pending_signal.update({"signal": None, "since": None, "last_retry": None})
                if AUTO_CLOSE_AT_SESSION_END:
                    position = _get_position(MT5_SYMBOL)
                    if position:
                        _close_position(position)
                continue

except KeyboardInterrupt:
    log("[INFO] Stopped by user (CTRL-C).")
finally:
    try:
        mt5.shutdown()
        log("[INFO] MT5 shutdown complete.")
    except Exception as e:
        log(f"[WARN] MT5 shutdown error: {e}")
