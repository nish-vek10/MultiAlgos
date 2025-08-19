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

# =========================
# USER CONFIG
# =========================

# --- Accounts / API ---
MT5_LOGIN         = 52474888
MT5_PASSWORD      = "2832ZeoEz!bZ$m"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\52474888\terminal64.exe"

OANDA_TOKEN       = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
OANDA_API_URL     = "https://api-fxpractice.oanda.com/v3"

# --- Timezone ---
LOCAL_TZ = timezone("Europe/London")

# --- Strategy parameters (same as your backtest spirit) ---
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 50
ATR_PERIOD      = 14
RISK_PERCENT    = 0.0025   # 0.25% of balance per entry (if RISK_MODE == "per_trade")
SL_MULTIPLIER   = 1.5
TP_MULTIPLIER   = 2.0

# --- Execution / risk settings ---
SLIPPAGE_POINTS   = 10
MIN_SECONDS_BETWEEN_ORDERS = 1
RISK_MODE                   = "per_trade"  # "per_trade" or "fixed"
FIXED_LOT_SIZE              = 0.10         # used if RISK_MODE == "fixed"
ALLOW_MULTIPLE_PER_BAR      = False        # 1 decision per completed bar (per asset)
PAUSE_DURING_ROLLOVER       = True         # 22:00–22:10 UTC safety pause
GRANULARITY                 = "M5"
NUM_CANDLES                 = 800          # history window

# --- Per-asset profiles (edit symbols/sessions here) ---
ASSETS = {
    "XAU": {
        "mt5_symbol": "XAUUSD",
        "oanda_symbol": "XAU_USD",
        "sessions": [("06:00", "12:00"), ("13:00", "18:00")],
        "magic": 888101,
        "max_total_open": 5,
        "max_per_side":  4,
        "auto_close_at_session_end": True,
        "close_all_magics_at_session_end": True,
    },
    "BTC": {
        "mt5_symbol": "BTCUSD",
        "oanda_symbol": "BTC_USD",
        "sessions": [("00:30", "21:00")],
        "magic": 888102,
        "max_total_open": 5,
        "max_per_side":  4,
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
    url = f"{OANDA_API_URL}/instruments/{oanda_symbol}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
    except Exception as e:
        print("[ERROR] OANDA request error:", str(e))
        return None
    if resp.status_code != 200:
        print("[ERROR] OANDA fetch failed:", resp.status_code, resp.text[:200])
        return None
    raw = resp.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for c in raw:
        if c.get("complete", False):
            utc = pd.to_datetime(c["time"], utc=True)
            lt = utc.tz_convert(LOCAL_TZ)
            data["time"].append(lt)
            data["open"].append(float(c["mid"]["o"]))
            data["high"].append(float(c["mid"]["h"]))
            data["low"].append(float(c["mid"]["l"]))
            data["close"].append(float(c["mid"]["c"]))
            data["volume"].append(int(c["volume"]))
    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index("time", inplace=True)
    return df

# =========================
# INDICATORS & SIGNALS
# =========================

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
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
            df = calc_indicators(df)
            if len(df) < max(EMA_SLOW_PERIOD, ATR_PERIOD) + 2:
                continue

            row_now  = df.iloc[-1]
            row_prev = df.iloc[-2]
            sig = generate_signal(row_now, row_prev)

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

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping EA (keyboard interrupt).")
        break
    except Exception as e:
        print("[EXCEPTION]", type(e).__name__, str(e))
        time.sleep(2)

# Shutdown
mt5.shutdown()
