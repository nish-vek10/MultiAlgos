# ===== EMA+VWAP Scalp — Multi-Asset (MT5 Live/Demo) =====
# Signals from OANDA completed candles; trades placed on MT5.
# Session-anchored VWAP, HTF EMA(50/200) filter, ATR regime, spread gate,
# SL = k*ATR, TP = m*SL, then breakeven -> trailing stop (ratchet).
# Per-asset parameters for TF/sessions/risk/SLTP/gates/trailing.

import os, sys, time, platform, struct
from datetime import datetime, timedelta, timezone as dt_timezone
import numpy as np
import pandas as pd
from pytz import timezone
import MetaTrader5 as mt5

# --- Your utils (assumed available in your project) ---
from utils.oanda_fetch import make_oanda_session, fetch_candles

# =========================
# USER CONFIG (EDIT THESE)
# =========================

# --- Accounts / API ---
MT5_LOGIN         = 52599744
MT5_PASSWORD      = "Tv95BcUCl!7fYj"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\52599744\terminal64.exe"

OANDA_TOKEN       = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_sess = make_oanda_session(OANDA_TOKEN, host="https://api-fxpractice.oanda.com", timeout=(3.05, 10))

# --- Timezone & indicator lengths ---
LOCAL_TZ = timezone("Europe/London")
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 50
ATR_PERIOD      = 14

# --- Global execution ---
SLIPPAGE_POINTS               = 10
RISK_MODE                     = "per_trade"   # "per_trade" or "fixed"
FIXED_LOT_SIZE                = 0.10          # used only if RISK_MODE == "fixed"
ALLOW_MULTIPLE_PER_BAR        = False         # 1 decision per completed bar (per asset)
PAUSE_DURING_ROLLOVER         = True          # 22:00–22:10 UTC safety pause
DEFAULT_GRANULARITY           = "M5"
NUM_CANDLES                   = 800
MIN_SECONDS_BETWEEN_ORDERS    = 1

print(f"[BOOT] MT5 path exists? {os.path.exists(MT5_TERMINAL_PATH)}")

# =========================
# PER-ASSET PROFILES
# =========================
# NOTE: Check mt5_symbol names broker (e.g., GER40.cash vs DE40.cash).
#       Check oanda_symbol names in OANDA practice account.

ASSETS = {
    # ===== DE40 / GER40 =====
    "DAX40": {
        "mt5_symbol": "DE40",
        "oanda_symbol": "DE30_EUR",
        "granularity": "M5",
        "sessions": [("07:00", "11:30"), ("13:00", "16:30")],
        "magic": 888201,
        "max_total_open": 4,
        "max_per_side":  3,
        "risk_percent": 0.0020,
        "sl_mult": 1.6,
        "tp_mult": 2.8,
        "max_spread_points": 15,
        "atr_pctl_low": 30, "atr_pctl_high": 85,
        "breakeven_at_r": 0.8,
        "trail_start_r": 1.3,
        "trail_atr_mult": 0.9,
        "auto_close_at_session_end": True,
        "close_all_magics_at_session_end": True,
    },

    # ===== NAS100 =====
    "NASDAQ": {
        "mt5_symbol": "USTEC",
        "oanda_symbol": "NAS100_USD",
        "granularity": "M5",
        "sessions": [("13:30", "17:00")],
        "magic": 888301,
        "max_total_open": 4,
        "max_per_side":  3,
        "risk_percent": 0.0015,
        "sl_mult": 1.8,
        "tp_mult": 3.0,
        "max_spread_points": 40,
        "atr_pctl_low": 40, "atr_pctl_high": 80,
        "breakeven_at_r": 0.9,
        "trail_start_r": 1.4,
        "trail_atr_mult": 1.0,
        "auto_close_at_session_end": True,
        "close_all_magics_at_session_end": True,
    },

    # ===== BTCUSD (consider isolating account) =====
    "BTC": {
        "mt5_symbol": "BTCUSD",
        "oanda_symbol": "BTC_USD",
        "granularity": "M5",
        "sessions": [("08:00", "11:00"), ("13:30", "16:00")],
        "magic": 888401,
        "max_total_open": 3,
        "max_per_side":  2,
        "risk_percent": 0.0010,
        "sl_mult": 1.8,
        "tp_mult": 2.0,
        "max_spread_points": 200,  # fallback ceiling; also uses adaptive rule
        "atr_pctl_low": 40, "atr_pctl_high": 75,
        "breakeven_at_r": 1.0,
        "trail_start_r": 1.5,
        "trail_atr_mult": 1.2,
        "auto_close_at_session_end": True,
        "close_all_magics_at_session_end": True,
    },

    # ===== EURUSD (stability baseline, M15) =====
    "EURUSD": {
        "mt5_symbol": "EURUSD",
        "oanda_symbol": "EUR_USD",
        "granularity": "M15",
        "sessions": [("07:00", "10:30"), ("13:00", "16:00")],
        "magic": 888501,
        "max_total_open": 3,
        "max_per_side":  2,
        "risk_percent": 0.0025,
        "sl_mult": 1.2,
        "tp_mult": 2.0,
        "max_spread_points": 25,  # broker 'point' may be fractional pip
        "atr_pctl_low": 25, "atr_pctl_high": 75,
        "breakeven_at_r": 1.0,
        "trail_start_r": 1.2,
        "trail_atr_mult": 0.8,
        "auto_close_at_session_end": True,
        "close_all_magics_at_session_end": True,
    },
}

print(f"[BOOT] Assets: " + ", ".join([f"{k}({v['mt5_symbol']}/{v['oanda_symbol']})" for k,v in ASSETS.items()]))

# =========================
# INIT MT5 (robust)
# =========================
def ensure_64bit():
    bits = struct.calcsize("P") * 8
    if bits != 64:
        raise RuntimeError(f"Python must be 64-bit for terminal64.exe (found {bits}-bit).")

def init_mt5():
    ensure_64bit()
    try:
        mt5.shutdown()
    except Exception:
        pass

    if os.path.exists(MT5_TERMINAL_PATH):
        ok = mt5.initialize(path=MT5_TERMINAL_PATH)
    else:
        print(f"[WARN] Terminal path not found: {MT5_TERMINAL_PATH} — trying default attach")
        ok = mt5.initialize()

    if not ok:
        print("[ERROR] mt5.initialize failed:", mt5.last_error())
        time.sleep(2.0)
        ok = mt5.initialize()
        if not ok:
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    info = mt5.account_info()
    if info is None or info.login != MT5_LOGIN:
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    ver = mt5.version()
    term = mt5.terminal_info()
    print(f"[MT5] API version: {ver}")
    print(f"[MT5] Terminal: build={getattr(term,'build',None)} pid={getattr(term,'pid',None)} data_path={getattr(term,'data_path',None)}")

    acct = mt5.account_info()
    if acct is None:
        raise RuntimeError(f"Failed to retrieve MT5 account info: {mt5.last_error()}")
    print(f"\n[MT5] Connected: {acct.login} | Balance: ${acct.balance:.2f}\n")
    return acct

acct = init_mt5()

# =========================
# SYMBOL INFO CACHE & HELPERS
# =========================
SYMBOL_INFOS = {}
def get_symbol_info(sym):
    info = mt5.symbol_info(sym)
    if info is None:
        raise RuntimeError(f"Symbol not found: {sym}")
    if not info.visible:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
        if info is None:
            raise RuntimeError(f"Cannot select symbol: {sym}")
    return info

def fmt_price_for(sym, x: float) -> str:
    d = SYMBOL_INFOS[sym]["digits"]
    return f"{float(x):.{d}f}"

def round_price_for(sym, x: float) -> float:
    d = SYMBOL_INFOS[sym]["digits"]
    return round(float(x), d)

def round_lots_for(sym, lots: float) -> float:
    info = SYMBOL_INFOS[sym]["info"]
    step = max(info.volume_step, 0.01)
    rounded = round(round(lots / step) * step, 2)
    return max(info.volume_min, min(rounded, info.volume_max))

def enforce_stop_level(sym, order_type, price, sl_price, tp_price):
    info = SYMBOL_INFOS[sym]["info"]
    stop_level_points = getattr(info, "trade_stops_level", 0)
    if stop_level_points and stop_level_points > 0:
        min_dist = stop_level_points * info.point
        if order_type == mt5.ORDER_TYPE_BUY:
            if price - sl_price < min_dist:
                sl_price = price - min_dist
            if tp_price - price < min_dist:
                tp_price = price + min_dist
        else:
            if sl_price - price < min_dist:
                sl_price = price + min_dist
            if price - tp_price < min_dist:
                tp_price = price - min_dist
    sl_price = round_price_for(sym, sl_price)
    tp_price = round_price_for(sym, tp_price)
    return sl_price, tp_price

for key, prof in ASSETS.items():
    info = get_symbol_info(prof["mt5_symbol"])
    SYMBOL_INFOS[prof["mt5_symbol"]] = {
        "info": info,
        "digits": info.digits,
        "point": info.point,
    }
    print(f"[MT5] {prof['mt5_symbol']} lot: min={info.volume_min}, max={info.volume_max}, "
          f"step={info.volume_step}, contract_size={info.trade_contract_size}, digits={info.digits}")

# =========================
# DATA & INDICATORS
# =========================
def fetch_oanda_candles_safe(oanda_symbol, granularity, count=NUM_CANDLES):
    try:
        j = fetch_candles(oanda_sess, instrument=oanda_symbol, granularity=granularity, count=count, price="M")
    except Exception as e:
        print("[ERROR] OANDA fetch:", str(e)[:200])
        return None
    raw = j.get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for c in raw:
        if c.get("complete", False):
            lt = pd.to_datetime(c["time"], utc=True).tz_convert(LOCAL_TZ)
            data["time"].append(lt)
            data["open"].append(float(c["mid"]["o"]))
            data["high"].append(float(c["mid"]["h"]))
            data["low"].append(float(c["mid"]["l"]))
            data["close"].append(float(c["mid"]["c"]))
            data["volume"].append(int(c["volume"]))
    if not data["time"]:
        return None
    return pd.DataFrame(data).set_index("time")

def session_vwap(df, sessions, tz):
    v = df.copy()
    local = v.index.tz_convert(tz)

    # build per-day sessions robustly
    sess_id = np.zeros(len(v), dtype=int)
    sid = 0

    # iterate unique local days (keeps tz)
    for day in pd.unique(local.normalize()):
        for s, e in sessions:
            sid += 1
            h_s, m_s = map(int, s.split(":"))
            h_e, m_e = map(int, e.split(":"))
            s_dt = day + pd.Timedelta(hours=h_s, minutes=m_s)
            e_dt = day + pd.Timedelta(hours=h_e, minutes=m_e)

            mask = (local >= s_dt) & (local < e_dt)   # <-- this is a NumPy bool array
            sess_id[mask] = sid                        # <-- use it directly

    # store with index to be safe for groupby alignment
    v["sess_id"] = pd.Series(sess_id, index=v.index)

    typical = (v["high"] + v["low"] + v["close"]) / 3.0
    vol = v["volume"].replace(0, np.nan).ffill()

    num = (typical * vol).groupby(v["sess_id"]).cumsum()
    den = vol.groupby(v["sess_id"]).cumsum()
    v["vwap"] = num / den
    return v["vwap"]

def calc_indicators(df: pd.DataFrame, sessions=None) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    if sessions:
        df["vwap"] = session_vwap(df, sessions, LOCAL_TZ)
    else:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, np.nan).ffill()
        df["vwap"] = (typical * vol).cumsum() / vol.cumsum()
    c_prev = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - c_prev).abs()
    tr3 = (df["low"] - c_prev).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    return df

def generate_signal(row_now: pd.Series, row_prev: pd.Series):
    need = ["vwap", "ema_fast", "ema_slow", "atr", "close", "low", "high"]
    if any(pd.isna(row_now.get(k)) for k in need):
        return None
    if (row_now["close"] > row_now["vwap"] and
        row_now["ema_fast"] > row_now["ema_slow"] and
        row_now["close"] > row_now["ema_fast"] and
        (row_prev["low"] <= row_now["ema_fast"])):
        return "BUY"
    if (row_now["close"] < row_now["vwap"] and
        row_now["ema_fast"] < row_now["ema_slow"] and
        row_now["close"] < row_now["ema_fast"] and
        (row_prev["high"] >= row_now["ema_fast"])):
        return "SELL"
    return None

# =========================
# SESSION HELPERS
# =========================
def _make_dt(base_dt, hhmm, tz):
    h, m = map(int, hhmm.split(":"))
    return base_dt.astimezone(tz).replace(hour=h, minute=m, second=0, microsecond=0)

def in_session(asset_key: str, now_local) -> bool:
    for start, end in ASSETS[asset_key]["sessions"]:
        s = _make_dt(now_local, start, LOCAL_TZ)
        e = _make_dt(now_local, end,   LOCAL_TZ)
        if s <= now_local < e:
            return True
    return False

# =========================
# TRADING HELPERS
# =========================
def get_open_positions(mt5_symbol: str, magic: int):
    poss = mt5.positions_get(symbol=mt5_symbol)
    if not poss:
        return []
    return [p for p in poss if p.magic == magic]

def count_positions_by_side(positions):
    buy = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_BUY)
    sell = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_SELL)
    return buy, sell

def choose_lot_size(mt5_symbol: str, balance: float, stop_distance_price: float, risk_percent: float):
    info = SYMBOL_INFOS[mt5_symbol]["info"]
    if RISK_MODE.lower() == "fixed":
        return round_lots_for(mt5_symbol, FIXED_LOT_SIZE)
    if stop_distance_price <= 0:
        return round_lots_for(mt5_symbol, info.volume_min)
    risk_dollars = balance * risk_percent
    lots_raw = risk_dollars / (stop_distance_price * info.trade_contract_size)
    return round_lots_for(mt5_symbol, lots_raw)

def place_order(mt5_symbol: str, direction: str, sl_price: float, tp_price: float, lots: float, magic: int, comment: str):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        print(f"[ERROR] ({mt5_symbol}) No tick data; cannot place order.")
        return False
    raw_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    price = round_price_for(mt5_symbol, raw_price)
    sl_price = round_price_for(mt5_symbol, sl_price)
    tp_price = round_price_for(mt5_symbol, tp_price)
    sl_price, tp_price = enforce_stop_level(mt5_symbol, order_type, price, sl_price, tp_price)
    lots = round_lots_for(mt5_symbol, lots)

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": SLIPPAGE_POINTS,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None:
        print(f"[ERROR] ({mt5_symbol}) order_send() None. Last error: {mt5.last_error()}")
        return False
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] ({mt5_symbol}) Order failed: retcode={res.retcode} comment={res.comment}")
        return False
    print(f"[OK] ({mt5_symbol}) Placed {direction} {round_lots_for(mt5_symbol, lots):.2f} @ {fmt_price_for(mt5_symbol, price)} | "
          f"SL={fmt_price_for(mt5_symbol, sl_price)} TP={fmt_price_for(mt5_symbol, tp_price)}")
    return True

def close_all_positions_for_asset(mt5_symbol: str, magic: int | None, max_rounds: int = 3, pause_s: float = 0.4):
    for attempt in range(1, max_rounds + 1):
        poss = mt5.positions_get(symbol=mt5_symbol)
        if not poss:
            return
        if magic is not None:
            poss = [p for p in poss if p.magic == magic]
        if not poss:
            return
        print(f"[AUTO-CLOSE] ({mt5_symbol}) Round {attempt}: closing {len(poss)} positions...")
        for p in poss:
            action = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                print(f"[ERROR] ({mt5_symbol}) No tick to close ticket={p.ticket}")
                continue
            price = tick.bid if action == mt5.ORDER_TYPE_SELL else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": p.volume,
                "type": action,
                "position": p.ticket,
                "price": price,
                "deviation": SLIPPAGE_POINTS,
                "magic": p.magic,
                "comment": "AutoClose_SessionEnd",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[ERROR] ({mt5_symbol}) Close failed: ticket={p.ticket} "
                      f"retcode={getattr(res,'retcode',None)} comment={getattr(res,'comment',None)}")
            else:
                print(f"[OK] ({mt5_symbol}) Auto-closed ticket={p.ticket}")
        time.sleep(pause_s)

def spread_ok(mt5_symbol, max_points, adaptive=False, oanda_df=None):
    info = mt5.symbol_info(mt5_symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)
    if not info or not tick:
        return False
    spread_points = (tick.ask - tick.bid) / info.point
    if adaptive and oanda_df is not None and len(oanda_df) >= 12:
        # adaptive: allow up to 2x median micro-range from last ~1h bars
        micro = (oanda_df["high"].tail(12) - oanda_df["low"].tail(12)).median()
        if micro > 0:
            micro_pts = micro / info.point
            return spread_points <= max(max_points, 2.0 * micro_pts)
    return spread_points <= max_points

def atr_regime_ok(df, p_low, p_high):
    atr_series = df["atr"].dropna()
    if len(atr_series) < 100:
        return True
    lo = np.nanpercentile(atr_series, p_low)
    hi = np.nanpercentile(atr_series, p_high)
    cur = atr_series.iloc[-1]
    return (cur >= lo) and (cur <= hi)

def htf_alignment_ok(oanda_symbol, direction):
    # M15 trend filter using EMA50 vs EMA200 across all assets
    try:
        df15 = fetch_oanda_candles(oanda_sess, instrument=oanda_symbol, granularity="M15", count=240)
    except Exception:
        return False
    if df15 is None or len(df15) < 210:
        return False
    ema50  = df15["close"].ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = df15["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    return (ema50 > ema200) if direction == "BUY" else (ema50 < ema200)

def modify_position_sl_tp(mt5_symbol, ticket, new_sl=None, new_tp=None):
    pos = next((p for p in (mt5.positions_get(symbol=mt5_symbol) or []) if p.ticket == ticket), None)
    if pos is None:
        return False
    sl = round_price_for(mt5_symbol, new_sl) if new_sl is not None else pos.sl
    tp = round_price_for(mt5_symbol, new_tp) if new_tp is not None else pos.tp
    if (sl == pos.sl) and (tp == pos.tp):
        return True
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": mt5_symbol,
        "position": ticket,
        "sl": sl,
        "tp": tp,
        "magic": pos.magic,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    res = mt5.order_send(req)
    return (res is not None) and (res.retcode == mt5.TRADE_RETCODE_DONE)

def enforce_min_stop_distance(mt5_symbol, order_type, price, sl_price):
    info = SYMBOL_INFOS[mt5_symbol]["info"]
    stop_level_points = getattr(info, "trade_stops_level", 0)
    if stop_level_points and stop_level_points > 0:
        min_dist = stop_level_points * info.point
        if order_type == mt5.ORDER_TYPE_BUY:
            sl_price = min(sl_price, price - min_dist)
        else:
            sl_price = max(sl_price, price + min_dist)
    return round_price_for(mt5_symbol, sl_price)

def manage_open_positions(mt5_symbol, magic, latest_atr, breakeven_at_r, trail_start_r, trail_atr_mult):
    info = SYMBOL_INFOS[mt5_symbol]["info"]
    tick = mt5.symbol_info_tick(mt5_symbol)
    if not tick:
        return
    poss = mt5.positions_get(symbol=mt5_symbol) or []
    mine = [p for p in poss if p.magic == magic]

    for p in mine:
        R_price = abs(p.price_open - p.sl) if p.sl not in (0.0, None) else 0.0
        if R_price <= 0:
            continue
        if p.type == mt5.ORDER_TYPE_BUY:
            cur_px = tick.bid
            r_mult = (cur_px - p.price_open) / R_price
        else:
            cur_px = tick.ask
            r_mult = (p.price_open - cur_px) / R_price

        # Breakeven
        if r_mult >= breakeven_at_r and round(p.sl, info.digits) != round(p.price_open, info.digits):
            new_sl = enforce_min_stop_distance(mt5_symbol, p.type, cur_px, p.price_open)
            modify_position_sl_tp(mt5_symbol, p.ticket, new_sl=new_sl)

        # Trailing
        if r_mult >= trail_start_r:
            trail_dist = trail_atr_mult * float(latest_atr)
            if p.type == mt5.ORDER_TYPE_BUY:
                target_sl = cur_px - trail_dist
                new_sl = max(target_sl, p.sl)
            else:
                target_sl = cur_px + trail_dist
                new_sl = min(target_sl, p.sl)
            new_sl = enforce_min_stop_distance(mt5_symbol, p.type, cur_px, new_sl)
            # Avoid micro-adjust spam
            if abs(new_sl - p.sl) >= info.point * max(2, info.trade_stops_level or 2):
                modify_position_sl_tp(mt5_symbol, p.ticket, new_sl=new_sl)

# =========================
# 5-MINUTE / 15-MINUTE SCHEDULING
# =========================
def _next_close(now, step_minutes=5):
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (step_minutes - (base.minute % step_minutes)) % step_minutes
    target = base + timedelta(minutes=mins_to_add)
    if target <= now:
        target += timedelta(minutes=step_minutes)
    return target

def countdown_to_next_close(tz, step_minutes=5):
    target = _next_close(datetime.now(tz), step_minutes)
    while True:
        now = datetime.now(tz)
        remain = (target - now).total_seconds()
        if remain <= 0:
            break
        time.sleep(0.25)
    time.sleep(2)  # finalize candle
    return target

def maybe_pause_for_rollover():
    if not PAUSE_DURING_ROLLOVER:
        return
    now_utc = datetime.now(dt_timezone.utc)
    if now_utc.hour == 22 and 0 <= now_utc.minute <= 10:
        print("[ROLLOVER] Pausing (22:00–22:10 UTC).")
        time.sleep(60)

# =========================
# MAIN LOOP
# =========================
state = {
    key: {
        "last_candle_time": None,
        "last_signal_bar": None,
        "last_order_time": 0.0,
        "in_session_prev": False,
    } for key in ASSETS.keys()
}

print("[RUN] Multi-asset EMA+VWAP scalper started.")

while True:
    try:
        maybe_pause_for_rollover()

        # Choose a global "next close" that suits the fastest asset (5m). EURUSD (15m) just uses per-asset granularity on fetch.
        next_close = countdown_to_next_close(LOCAL_TZ, step_minutes=5)

        for key, prof in ASSETS.items():
            now_local = datetime.now(LOCAL_TZ)
            curr_in_session = in_session(key, now_local)

            close_all_magics = prof.get("close_all_magics_at_session_end", False)
            magic_filter = None if close_all_magics else prof["magic"]

            if state[key]["in_session_prev"] and not curr_in_session and prof.get("auto_close_at_session_end", False):
                print(f"[SESSION] ({prof['mt5_symbol']}) Session ended — auto-closing open positions...")
                close_all_positions_for_asset(prof["mt5_symbol"], magic_filter)

            if not curr_in_session and prof.get("auto_close_at_session_end", False):
                existing = mt5.positions_get(symbol=prof["mt5_symbol"]) or []
                relevant = existing if magic_filter is None else [p for p in existing if p.magic == magic_filter]
                if relevant:
                    print(f"[SESSION] ({prof['mt5_symbol']}) Out of session — ensuring flat...")
                    close_all_positions_for_asset(prof["mt5_symbol"], magic_filter)

            state[key]["in_session_prev"] = curr_in_session
            if not curr_in_session:
                continue

            mt5_symbol   = prof["mt5_symbol"]
            oanda_symbol = prof["oanda_symbol"]
            magic        = prof["magic"]
            cap_total    = prof["max_total_open"]
            cap_side     = prof["max_per_side"]

            granularity  = prof.get("granularity", DEFAULT_GRANULARITY)
            risk_percent = prof["risk_percent"]
            sl_mult      = prof["sl_mult"]
            tp_mult      = prof["tp_mult"]
            max_spread   = prof["max_spread_points"]
            p_low        = prof["atr_pctl_low"]
            p_high       = prof["atr_pctl_high"]
            be_r         = prof["breakeven_at_r"]
            trail_r      = prof["trail_start_r"]
            trail_mult   = prof["trail_atr_mult"]

            s = state[key]
            if s["last_candle_time"] is not None and s["last_candle_time"] >= (next_close - timedelta(minutes=5)):
                continue

            # Fetch until a new bar
            retry_timeout = timedelta(seconds=45)
            start_poll = datetime.now(LOCAL_TZ)
            df = None
            prev_anchor = s["last_candle_time"]

            while True:
                if not in_session(key, datetime.now(LOCAL_TZ)):
                    df = None
                    break
                df = fetch_oanda_candles_safe(oanda_symbol, granularity, NUM_CANDLES)
                if df is None or df.empty:
                    time.sleep(2)
                    if datetime.now(LOCAL_TZ) - start_poll > retry_timeout:
                        df = None
                        break
                    continue
                newest_time = df.index[-1]
                if prev_anchor is None or newest_time > prev_anchor:
                    s["last_candle_time"] = newest_time
                    break
                time.sleep(2)
                if datetime.now(LOCAL_TZ) - start_poll > retry_timeout:
                    df = None
                    break

            if df is None or df.empty:
                continue

            # Indicators & signal
            df = calc_indicators(df, sessions=ASSETS[key]["sessions"])
            if len(df) < max(EMA_SLOW_PERIOD, ATR_PERIOD) + 2:
                continue

            row_now  = df.iloc[-1]
            row_prev = df.iloc[-2]
            sig = generate_signal(row_now, row_prev)

            # ---- Quality gates ----
            if key == "BTC":
                if not spread_ok(mt5_symbol, max_spread, adaptive=True, oanda_df=df):
                    continue
            else:
                if not spread_ok(mt5_symbol, max_spread):
                    continue

            if not atr_regime_ok(df, p_low, p_high):
                continue

            if sig in ("BUY", "SELL") and not htf_alignment_ok(oanda_symbol, sig):
                continue

            if not in_session(key, datetime.now(LOCAL_TZ)):
                continue

            # Position state (filtered to our magic)
            positions = get_open_positions(mt5_symbol, magic)
            buy_count, sell_count = count_positions_by_side(positions)
            total_open = len(positions)

            current_bar_ts = df.index[-1]
            if not ALLOW_MULTIPLE_PER_BAR and sig in ("BUY", "SELL") and s["last_signal_bar"] == current_bar_ts:
                sig = None

            # Trade
            if sig in ("BUY", "SELL"):
                atr   = float(row_now["atr"])
                price = float(row_now["close"])
                if not np.isnan(atr) and atr > 0:
                    stop_distance = sl_mult * atr
                    if sig == "BUY":
                        sl_price = price - stop_distance
                        tp_price = price + tp_mult * stop_distance
                    else:
                        sl_price = price + stop_distance
                        tp_price = price - tp_mult * stop_distance

                    side_count   = buy_count if sig == "BUY" else sell_count
                    can_add_side = side_count < cap_side
                    can_add_total= total_open  < cap_total

                    now_ts = time.time()
                    if (now_ts - s["last_order_time"] >= MIN_SECONDS_BETWEEN_ORDERS) and can_add_total and can_add_side:
                        lots = choose_lot_size(mt5_symbol, mt5.account_info().balance, stop_distance, risk_percent)
                        ok = place_order(mt5_symbol, sig, sl_price, tp_price, lots, magic, f"EMA_VWAP_{key}")
                        if ok:
                            s["last_order_time"] = time.time()
                            s["last_signal_bar"] = current_bar_ts

            # Manage breakeven/trailing each bar
            latest_atr = float(row_now["atr"]) if not np.isnan(row_now["atr"]) else None
            if latest_atr:
                manage_open_positions(mt5_symbol, magic, latest_atr, be_r, trail_r, trail_mult)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping EA (keyboard interrupt).")
        break
    except Exception as e:
        print("[EXCEPTION]", type(e).__name__, str(e))
        time.sleep(2)

# Shutdown
mt5.shutdown()
