"""
EMA+VWAP Scalp — Multi-Asset (XAUUSD + BTCUSD) — MT5 Live/Demo
- Uses OANDA 5m completed candles for signals
- Trades on MT5 with SL/TP attached
- Session-gated per asset (London time windows)
- Optional rollover pause (22:00–22:10 UTC)
- Multiple concurrent entries allowed with caps, per asset
"""

# ===== FILENAME: EMA-VWAP_Scalp_Multi =====

import sys
import time
from datetime import datetime, timedelta, timezone as dt_timezone
import requests
import pandas as pd
import numpy as np
from pytz import timezone
import MetaTrader5 as mt5
from utils.oanda_fetch import make_oanda_session, fetch_candles

# =========================
# USER CONFIG
# =========================

# --- Accounts / API ---
MT5_LOGIN         = 52583222
MT5_PASSWORD      = "98qJk@gb2qmlDP"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\52583222\terminal64.exe"

OANDA_TOKEN       = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
OANDA_API_URL     = "https://api-fxpractice.oanda.com/v3"
oanda_sess = make_oanda_session(OANDA_TOKEN, host="https://api-fxpractice.oanda.com", timeout=(3.05, 10))

# --- Timezone ---
LOCAL_TZ = timezone("Europe/London")

# --- Strategy parameters (same as your backtest spirit) ---
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 50
ATR_PERIOD      = 14
RISK_PERCENT    = 0.0025   # 0.25% of balance per entry (if RISK_MODE == "per_trade")
SL_MULTIPLIER   = 1.5
TP_MULTIPLIER   = 2.6

# --- Execution / risk settings ---
SLIPPAGE_POINTS   = 10
MIN_SECONDS_BETWEEN_ORDERS = 1
RISK_MODE                   = "per_trade"  # "per_trade" or "fixed"
FIXED_LOT_SIZE              = 0.10         # used if RISK_MODE == "fixed"
ALLOW_MULTIPLE_PER_BAR      = False        # 1 decision per completed bar (per asset)
PAUSE_DURING_ROLLOVER       = True         # 22:00–22:10 UTC safety pause
GRANULARITY                 = "M5"
NUM_CANDLES                 = 800          # history window

# --- Trade-quality gating & trailing (XAU) ---
MAX_SPREAD_POINTS_XAU      = 30      # skip if spread > 30 points (~$0.30 if digits=2)
ATR_PCTL_LOW, ATR_PCTL_HIGH= 30, 80  # only trade when ATR in this percentile band
BREAKEVEN_AT_R             = 0.8     # move SL to entry at +0.8R
TRAIL_START_R              = 1.3     # start trailing at +1.3R
TRAIL_ATR_MULT             = 0.9     # trailing distance = 0.9*ATR (ratchet only)


# --- Per-asset profiles (edit symbols/sessions here) ---
ASSETS = {
    "XAU": {
        "mt5_symbol": "XAUUSD",
        "oanda_symbol": "XAU_USD",
        "sessions": [("06:00", "11:00"), ("13:30", "16:30")],
        "magic": 888101,
        "max_total_open": 4,
        "max_per_side":  3,
        "auto_close_at_session_end": True,
        "close_all_magics_at_session_end": True,
    },
}

print(f"[BOOT] MT5 path exists? {__import__('os').path.exists(MT5_TERMINAL_PATH)}")
print(f"[BOOT] Assets: " + ", ".join([f"{k}({v['mt5_symbol']}/{v['oanda_symbol']})" for k,v in ASSETS.items()]))

# =========================
# INIT MT5
# =========================

if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, path=MT5_TERMINAL_PATH):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    sys.exit(1)

acct = mt5.account_info()
if acct is None:
    raise RuntimeError(f"Failed to retrieve MT5 account info: {mt5.last_error()}")
print(f"\n[MT5] Connected: {acct.login} | Balance: ${acct.balance:.2f}\n")

# Cache per-asset symbol info and formatting helpers
SYMBOL_INFOS = {}
def get_symbol_info(sym):
    info = mt5.symbol_info(sym)
    if info is None:
        raise RuntimeError(f"Symbol not found: {sym}")
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
# OANDA DATA
# =========================

def fetch_oanda_candles(oanda_symbol, granularity=GRANULARITY, count=NUM_CANDLES):
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


# =========================
# INDICATORS & SIGNALS
# =========================

def session_vwap(df, sessions, tz):
    v = df.copy()
    local = v.index.tz_convert(tz)
    # Build a session id that increments at each session start (per day)
    sess_id = np.zeros(len(v), dtype=int)
    sid = 0
    for day in pd.to_datetime(pd.Series(local.date).unique()):
        for s, e in sessions:
            sid += 1
            s_dt = pd.Timestamp.combine(day, pd.to_datetime(s).time()).tz_localize(tz)
            e_dt = pd.Timestamp.combine(day, pd.to_datetime(e).time()).tz_localize(tz)
            mask = (local >= s_dt) & (local < e_dt)
            sess_id[mask.values] = sid
    v["sess_id"] = sess_id
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
    # session-anchored VWAP (fallback to legacy if sessions None)
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
    mine = [p for p in poss if p.magic == magic]
    return mine

def count_positions_by_side(positions):
    buy = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_BUY)
    sell = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_SELL)
    return buy, sell

def choose_lot_size(mt5_symbol: str, balance: float, stop_distance_price: float):
    info = SYMBOL_INFOS[mt5_symbol]["info"]
    if RISK_MODE.lower() == "fixed":
        return round_lots_for(mt5_symbol, FIXED_LOT_SIZE)
    if stop_distance_price <= 0:
        return round_lots_for(mt5_symbol, info.volume_min)
    risk_dollars = balance * RISK_PERCENT
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
    print(f"[OK] ({mt5_symbol}) Placed {direction} {round_lots_for(mt5_symbol, lots):.2f} lots @ {fmt_price_for(mt5_symbol, price)} | "
          f"SL={fmt_price_for(mt5_symbol, sl_price)} TP={fmt_price_for(mt5_symbol, tp_price)}")
    return True

# =========================
# 5-MINUTE SCHEDULING
# =========================

def _next_5m_close(now):
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    target = base + timedelta(minutes=mins_to_add)
    if target <= now:
        target += timedelta(minutes=5)
    return target

def countdown_to_next_5m(tz):
    target = _next_5m_close(datetime.now(tz))
    # muted countdown; sleeps till close
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

def close_all_positions_for_asset(mt5_symbol: str, magic: int | None, max_rounds: int = 3, pause_s: float = 0.4):
    """
    Close ALL open positions for a symbol, filtered by magic if provided.
    Retries up to max_rounds with a fresh re-read each round.
    """
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


def spread_ok(mt5_symbol):
    info = mt5.symbol_info(mt5_symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)
    if not info or not tick:
        return False
    spread_points = (tick.ask - tick.bid) / info.point
    return spread_points <= MAX_SPREAD_POINTS_XAU

def atr_regime_ok(df):
    atr_series = df["atr"].dropna()
    if len(atr_series) < 100:
        return True  # not enough data to judge; allow
    lo = np.nanpercentile(atr_series, ATR_PCTL_LOW)
    hi = np.nanpercentile(atr_series, ATR_PCTL_HIGH)
    cur = atr_series.iloc[-1]
    return (cur >= lo) and (cur <= hi)

def htf_alignment_ok(oanda_symbol, direction):
    # M15 trend filter using EMA50 vs EMA200
    try:
        df15 = fetch_oanda_candles(oanda_symbol, granularity="M15", count=240)
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
    info = SYMBOL_INFOS[mt5_symbol]["info"]
    # Round to symbol digits
    sl = round_price_for(mt5_symbol, new_sl) if new_sl is not None else pos.sl
    tp = round_price_for(mt5_symbol, new_tp) if new_tp is not None else pos.tp
    # If nothing changes, don't send a request
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

def manage_open_positions(mt5_symbol, magic, latest_atr):
    """
    - Move to breakeven at +0.8R
    - Start trailing at +1.3R with 0.9*ATR (ratchet only tighter)
    """
    info = SYMBOL_INFOS[mt5_symbol]["info"]
    tick = mt5.symbol_info_tick(mt5_symbol)
    if not tick:
        return
    poss = mt5.positions_get(symbol=mt5_symbol) or []
    mine = [p for p in poss if p.magic == magic]

    for p in mine:
        # R in price units from original SL
        R_price = abs(p.price_open - p.sl) if p.sl not in (0.0, None) else 0.0
        if R_price <= 0:
            continue
        # Current price and R multiple
        if p.type == mt5.ORDER_TYPE_BUY:
            cur_px = tick.bid
            r_mult = (cur_px - p.price_open) / R_price
        else:
            cur_px = tick.ask
            r_mult = (p.price_open - cur_px) / R_price

        # 1) Breakeven
        if r_mult >= BREAKEVEN_AT_R and round(p.sl, info.digits) != round(p.price_open, info.digits):
            new_sl = p.price_open
            new_sl = enforce_min_stop_distance(mt5_symbol, p.type, cur_px, new_sl)
            modify_position_sl_tp(mt5_symbol, p.ticket, new_sl=new_sl)

        # 2) Trailing
        if r_mult >= TRAIL_START_R:
            trail_dist = TRAIL_ATR_MULT * float(latest_atr)
            if p.type == mt5.ORDER_TYPE_BUY:
                target_sl = cur_px - trail_dist
                # never loosen the stop—ratchet only
                new_sl = max(target_sl, p.sl)
                new_sl = enforce_min_stop_distance(mt5_symbol, p.type, cur_px, new_sl)
            else:
                target_sl = cur_px + trail_dist
                new_sl = min(target_sl, p.sl)
                new_sl = enforce_min_stop_distance(mt5_symbol, p.type, cur_px, new_sl)

            # Avoid micro-adjust spam: only modify if change is meaningful
            if abs(new_sl - p.sl) >= info.point * max(2, info.trade_stops_level or 2):
                modify_position_sl_tp(mt5_symbol, p.ticket, new_sl=new_sl)


# =========================
# MAIN LOOP (multi-asset)
# =========================

# Per-asset state
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

        # 1) Wait for next global 5m close (one clock drives both assets)
        next_close = countdown_to_next_5m(LOCAL_TZ)

        # 2) Process each asset independently
        for key, prof in ASSETS.items():
            now_local = datetime.now(LOCAL_TZ)
            curr_in_session = in_session(key, now_local)

            # Determine whether we should close *all magics* or only ours
            close_all_magics = prof.get("close_all_magics_at_session_end", False)
            magic_filter = None if close_all_magics else prof["magic"]

            # Edge-trigger: just left session?
            if state[key]["in_session_prev"] and not curr_in_session and prof.get("auto_close_at_session_end", False):
                print(f"[SESSION] ({prof['mt5_symbol']}) Session ended — auto-closing open positions...")
                close_all_positions_for_asset(prof["mt5_symbol"], magic_filter)

            # Continuous guard: if we're out of session and auto-close is on, ensure flat
            if not curr_in_session and prof.get("auto_close_at_session_end", False):
                # only act if there are any positions to avoid spam:
                existing = mt5.positions_get(symbol=prof["mt5_symbol"]) or []
                if magic_filter is None:
                    relevant = existing
                else:
                    relevant = [p for p in existing if p.magic == magic_filter]
                if relevant:
                    print(f"[SESSION] ({prof['mt5_symbol']}) Out of session — ensuring flat...")
                    close_all_positions_for_asset(prof["mt5_symbol"], magic_filter)

            # Update the state now; we'll use curr_in_session below to gate entries.
            state[key]["in_session_prev"] = curr_in_session

            # Skip the rest of processing if out of session (prevents new entries)
            if not curr_in_session:
                continue

            if not in_session(key, now_local):
                # silent skip out-of-session
                continue

            mt5_symbol   = prof["mt5_symbol"]
            oanda_symbol = prof["oanda_symbol"]
            magic        = prof["magic"]
            cap_total    = prof["max_total_open"]
            cap_side     = prof["max_per_side"]
            s = state[key]

            # Avoid post-close noisy window if we already have the just-closed bar
            if s["last_candle_time"] is not None and s["last_candle_time"] >= (next_close - timedelta(minutes=5)):
                # wait to the next bar close; no extra logging to keep output tidy
                continue

            # Fetch data with a short retry until a new bar appears
            retry_timeout = timedelta(seconds=45)
            start_poll = datetime.now(LOCAL_TZ)
            df = None
            prev_anchor = s["last_candle_time"]

            while True:
                if not in_session(key, datetime.now(LOCAL_TZ)):
                    df = None
                    break
                df = fetch_oanda_candles(oanda_symbol, GRANULARITY, NUM_CANDLES)
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

            # ---- Quality gates (XAU) ----
            if not spread_ok(mt5_symbol):
                continue
            if not atr_regime_ok(df):
                continue
            if sig in ("BUY", "SELL") and not htf_alignment_ok(oanda_symbol, sig):
                continue

            # Re-check session before executing
            if not in_session(key, datetime.now(LOCAL_TZ)):
                continue

            # Position state (filtered to our magic)
            positions = get_open_positions(mt5_symbol, magic)
            buy_count, sell_count = count_positions_by_side(positions)
            total_open = len(positions)

            # (Optional) dedup per completed bar
            current_bar_ts = df.index[-1]
            if not ALLOW_MULTIPLE_PER_BAR and sig in ("BUY", "SELL") and s["last_signal_bar"] == current_bar_ts:
                sig = None

            # Trade
            if sig in ("BUY", "SELL"):
                atr   = float(row_now["atr"])
                price = float(row_now["close"])
                if pd.isna(atr) or atr <= 0:
                    pass
                else:
                    stop_distance = SL_MULTIPLIER * atr
                    if sig == "BUY":
                        sl_price = price - stop_distance
                        tp_price = price + TP_MULTIPLIER * stop_distance
                    else:
                        sl_price = price + stop_distance
                        tp_price = price - TP_MULTIPLIER * stop_distance

                    side_count   = buy_count if sig == "BUY" else sell_count
                    can_add_side = side_count < cap_side
                    can_add_total= total_open  < cap_total

                    now_ts = time.time()
                    if now_ts - s["last_order_time"] < MIN_SECONDS_BETWEEN_ORDERS:
                        pass
                    elif not can_add_total or not can_add_side:
                        pass
                    else:
                        lots = choose_lot_size(mt5_symbol, mt5.account_info().balance, stop_distance)
                        ok = place_order(mt5_symbol, sig, sl_price, tp_price, lots, magic, f"EMA_VWAP_{key}")
                        if ok:
                            s["last_order_time"] = time.time()
                            s["last_signal_bar"] = current_bar_ts

            # Run breakeven/trailing manager each bar with latest ATR
            latest_atr = float(row_now["atr"]) if not pd.isna(row_now["atr"]) else None
            if latest_atr:
                manage_open_positions(mt5_symbol, magic, latest_atr)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping EA (keyboard interrupt).")
        break
    except Exception as e:
        print("[EXCEPTION]", type(e).__name__, str(e))
        time.sleep(2)

# Shutdown
mt5.shutdown()
