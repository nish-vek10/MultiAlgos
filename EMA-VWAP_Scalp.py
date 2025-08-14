"""
EMA+VWAP Scalp for XAUUSD (5m) — Live/Demo MT5
- Uses OANDA 5m completed candles for signals
- Trades on MT5 with SL/TP attached
- 24/5 (no session gating), optional rollover pause
- Multiple concurrent entries allowed with caps (scalper behavior)
"""

# ===== FILENAME: EMA-VWAP_Scalp =====

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

# --- Symbols / timeframe ---
MT5_SYMBOL   = "XAUUSD"
OANDA_SYMBOL = "XAU_USD"
GRANULARITY  = "M5"
NUM_CANDLES  = 800  # history window for indicators

# --- Strategy parameters (same spirit as your backtest) ---
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 50
ATR_PERIOD      = 14
RISK_PERCENT    = 0.0025   # 0.25% of balance risk per entry (when RISK_MODE == "per_trade")
SL_MULTIPLIER   = 1.5
TP_MULTIPLIER   = 2.0

# --- Execution / risk settings ---
SLIPPAGE_POINTS   = 10
MAGIC_NUMBER      = 888888
MIN_SECONDS_BETWEEN_ORDERS = 1

# --- Scalping / re-entry controls ---
MAX_TOTAL_OPEN_TRADES       = 5     # hard cap for all open trades
MAX_OPEN_TRADES_PER_SIDE    = 3     # per direction cap
MIN_SECONDS_BETWEEN_ENTRIES = 5     # spacing between new entries
CLOSE_OPPOSITE_ON_SIGNAL    = False # if True: close opposite-side positions before new entry
RISK_MODE                   = "per_trade"  # "per_trade" or "fixed"
FIXED_LOT_SIZE              = 0.10         # used if RISK_MODE == "fixed"

# --- Time / misc ---
LOCAL_TZ = timezone("Europe/London")
PAUSE_DURING_ROLLOVER = True  # pause ~22:00–22:10 UTC
ALLOW_MULTIPLE_PER_BAR = False  # 1 decision per completed bar (safer for close-logic scalper)

print(f"[BOOT] Path exists? {__import__('os').path.exists(MT5_TERMINAL_PATH)}")

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

sym_info = mt5.symbol_info(MT5_SYMBOL)
if sym_info is None:
    raise RuntimeError(f"Symbol not found: {MT5_SYMBOL}")
print(f"[MT5] {MT5_SYMBOL} lot: min={sym_info.volume_min}, max={sym_info.volume_max}, "
      f"step={sym_info.volume_step}, contract_size={sym_info.trade_contract_size}")

# --- formatting helpers bound to the symbol's precision ---
SYMBOL_DIGITS = sym_info.digits          # price decimals (XAUUSD usually 2)
SYMBOL_POINT  = sym_info.point

def round_price(x: float) -> float:
    return round(float(x), SYMBOL_DIGITS)

def fmt_price(x: float) -> str:
    return f"{float(x):.{SYMBOL_DIGITS}f}"

def round_volume_2dp(lots: float) -> float:
    # respect broker step but clamp to at least 0.01 and return 2dp
    step = max(sym_info.volume_step, 0.01)
    rounded = round(round(lots / step) * step, 2)
    # keep within broker bounds
    return max(sym_info.volume_min, min(rounded, sym_info.volume_max))

def fmt_lots(x: float) -> str:
    return f"{float(x):.2f}"

# --- broker stop-level enforcement ---
def enforce_stop_level(order_type, price, sl_price, tp_price):
    """
    Ensure SL/TP are at least 'trade_stops_level' points away from the price.
    Returns (sl_price, tp_price) possibly adjusted and rounded to symbol digits.
    """
    info = mt5.symbol_info(MT5_SYMBOL)
    stop_level_points = getattr(info, "trade_stops_level", 0)
    if stop_level_points and stop_level_points > 0:
        min_dist = stop_level_points * info.point
        if order_type == mt5.ORDER_TYPE_BUY:
            # SL must be <= price - min_dist; TP must be >= price + min_dist
            if price - sl_price < min_dist:
                sl_price = price - min_dist
            if tp_price - price < min_dist:
                tp_price = price + min_dist
        else:  # SELL
            if sl_price - price < min_dist:
                sl_price = price + min_dist
            if price - tp_price < min_dist:
                tp_price = price - min_dist
    return round_price(sl_price), round_price(tp_price)

# =========================
# OANDA DATA
# =========================

def fetch_oanda_candles(symbol=OANDA_SYMBOL, granularity=GRANULARITY, count=NUM_CANDLES):
    """
    Fetch completed candles from OANDA and convert timestamps to LOCAL_TZ.
    """
    url = f"{OANDA_API_URL}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"granularity": granularity, "count": count, "price": "M"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
    except Exception as e:
        print("[ERROR] OANDA request error:", str(e))
        return None

    if resp.status_code != 200:
        print("[ERROR] OANDA fetch failed:", resp.status_code, resp.text[:180])
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
# INDICATORS
# =========================

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()

    # VWAP (cumulative)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan).ffill()
    df["vwap"] = (typical * vol).cumsum() / vol.cumsum()

    # ATR (classic TR then simple mean; matches backtest intent)
    c_prev = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - c_prev).abs()
    tr3 = (df["low"] - c_prev).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    return df

# =========================
# TIMING HELPERS (5m bar)
# =========================

def _next_5m_close(now):
    base = now.replace(second=0, microsecond=0)
    mins_to_add = (5 - (base.minute % 5)) % 5
    target = base + timedelta(minutes=mins_to_add)
    # ensure the target is strictly in the future
    if target <= now:
        target += timedelta(minutes=5)
    return target

def countdown_to_next_5m(tz, until=None):
    target = _next_5m_close(datetime.now(tz))
    if until is not None and target > until:
        target = until
    print(f"\n[*] Waiting for 5m close at {target.strftime('%H:%M:%S %Z')} ...")
    while True:
        now = datetime.now(tz)
        remain = (target - now).total_seconds()
        if remain <= 0:
            break
        m, s = divmod(int(remain + 0.5), 60)
        # sys.stdout.write(f"\r    Time remaining: {m:02d}:{s:02d}")
        # sys.stdout.flush()
        time.sleep(0.25)
    # print("\r    Time remaining: 00:00")
    time.sleep(2)  # let the provider finalize candle
    return target

def countdown_to(target_dt, tz):
    print(f"\n[*] Waiting until {target_dt.strftime('%H:%M:%S %Z')} ...")
    while True:
        now = datetime.now(tz)
        remain = (target_dt - now).total_seconds()
        if remain <= 0:
            break
        m, s = divmod(int(remain + 0.5), 60)
        # sys.stdout.write(f"\r    Time remaining: {m:02d}:{s:02d}")
        # sys.stdout.flush()
        time.sleep(0.25)
    print("\r    Time remaining: 00:00")
    time.sleep(2)

def maybe_pause_for_rollover():
    """
    Optional safety: pause during the typical rollover spike window.
    Uses UTC clock (22:00–22:10 UTC).
    """
    if not PAUSE_DURING_ROLLOVER:
        return
    now_utc = datetime.now(dt_timezone.utc)   # timezone-aware
    if now_utc.hour == 22 and 0 <= now_utc.minute <= 10:
        print("[ROLLOVER] Pausing due to potential spread widening (22:00–22:10 UTC).")
        time.sleep(60)

# =========================
# TRADING HELPERS
# =========================

def get_open_positions(symbol: str):
    """Return list of open positions for symbol (filtered by our MAGIC_NUMBER if any exist)."""
    poss = mt5.positions_get(symbol=symbol)
    if not poss:
        return []
    mine = [p for p in poss if p.magic == MAGIC_NUMBER]
    return mine if mine else list(poss)

def count_positions_by_side(positions):
    buy = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_BUY)
    sell = sum(1 for p in positions if p.type == mt5.ORDER_TYPE_SELL)
    return buy, sell

def round_volume(lots: float, info) -> float:
    step = info.volume_step
    rounded = round(lots / step) * step
    return max(info.volume_min, min(rounded, info.volume_max))

def compute_position_size(balance: float, stop_distance_price: float, info) -> float:
    """
    Risk in $, stop distance in price units (USD per ounce on XAU).
    risk = stop_distance * contract_size * lots  -> lots = risk / (stop_distance * contract_size)
    """
    if stop_distance_price <= 0:
        return info.volume_min
    risk_dollars = balance * RISK_PERCENT
    lots = risk_dollars / (stop_distance_price * info.trade_contract_size)
    return round_volume(lots, info)

def choose_lot_size(balance: float, stop_distance_price: float, info):
    if RISK_MODE.lower() == "fixed":
        return round_volume_2dp(FIXED_LOT_SIZE)
    # per-trade risk
    if stop_distance_price <= 0:
        return round_volume_2dp(info.volume_min)
    risk_dollars = balance * RISK_PERCENT
    lots_raw = risk_dollars / (stop_distance_price * info.trade_contract_size)
    return round_volume_2dp(lots_raw)

def place_order(direction: str, entry_price: float, sl_price: float, tp_price: float, lots: float):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None:
        print("[ERROR] No tick data; cannot place order.")
        return False

    # round prices to symbol precision
    raw_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    price = round_price(raw_price)
    sl_price = round_price(sl_price)
    tp_price = round_price(tp_price)

    # enforce broker min stop distance
    sl_price, tp_price = enforce_stop_level(order_type, price, sl_price, tp_price)

    # enforce 2dp / step-valid lots
    lots = round_volume_2dp(lots)

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": SLIPPAGE_POINTS,
        "magic": MAGIC_NUMBER,
        "comment": "EMA_VWAP_Scalp",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None:
        print(f"[ERROR] order_send() returned None. Last error: {mt5.last_error()}")
        return False
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order failed: retcode={res.retcode} comment={res.comment}")
        return False

    print(f"[OK] Placed {direction} {fmt_lots(lots)} lots @ {fmt_price(price)} | "
          f"SL={fmt_price(sl_price)} TP={fmt_price(tp_price)}")
    return True

def close_position_market(pos):
    """Market-close the position at current price (used if we need to flatten/flip)."""
    action = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None:
        print("[ERROR] No tick to close position.")
        return False
    price = tick.bid if action == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": pos.volume,
        "type": action,
        "position": pos.ticket,
        "price": price,
        "deviation": SLIPPAGE_POINTS,
        "magic": MAGIC_NUMBER,
        "comment": "EMA_VWAP_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Close failed: {getattr(res, 'retcode', None)} {getattr(res, 'comment', None)}")
        return False
    print(f"[OK] Closed position ticket={pos.ticket}")
    return True

# =========================
# SIGNALS (same logic as backtest)
# =========================

def generate_signal(row_now: pd.Series, row_prev: pd.Series):
    """
    Entry rules (end-of-bar):
    LONG:
        close > vwap AND ema_fast > ema_slow AND close > ema_fast AND (prev low <= current ema_fast)
    SHORT:
        close < vwap AND ema_fast < ema_slow AND close < ema_fast AND (prev high >= current ema_fast)
    """
    needed = ["vwap", "ema_fast", "ema_slow", "atr", "close", "low", "high"]
    if any(pd.isna(row_now.get(k)) for k in needed):
        return None

    # Long
    if (row_now["close"] > row_now["vwap"] and
        row_now["ema_fast"] > row_now["ema_slow"] and
        row_now["close"] > row_now["ema_fast"] and
        (row_prev["low"] <= row_now["ema_fast"])):
        return "BUY"

    # Short
    if (row_now["close"] < row_now["vwap"] and
        row_now["ema_fast"] < row_now["ema_slow"] and
        row_now["close"] < row_now["ema_fast"] and
        (row_prev["high"] >= row_now["ema_fast"])):
        return "SELL"

    return None

# =========================
# MAIN LOOP (24/5)
# =========================

last_candle_time = None
last_order_time = 0.0
last_signal_bar = None  # to avoid multiple entries on the same completed bar if desired

while True:
    try:
        maybe_pause_for_rollover()

        # 1) Wait for next 5m close (no session gating)
        next_close = countdown_to_next_5m(LOCAL_TZ)

        # If we already have the candle that just closed (time-stamped at next_close - 5m),
        # there won't be a new timestamp until the *next* close. Avoid pointless polling.
        if last_candle_time is not None and last_candle_time >= (next_close - timedelta(minutes=5)):
            print("[SKIP] Just-closed candle already fetched. Waiting for the next close...")
            next_next_close = next_close + timedelta(minutes=5)
            countdown_to(next_next_close, LOCAL_TZ)

        # 2) Fetch data and require a truly new candle (short, calm retry)
        retry_timeout = timedelta(seconds=60)
        start_poll = datetime.now(LOCAL_TZ)
        df = None
        prev_anchor = last_candle_time

        while True:
            df = fetch_oanda_candles(OANDA_SYMBOL, GRANULARITY, NUM_CANDLES)
            if df is None or df.empty:
                print("[ERROR] No data returned; retrying shortly...")
                time.sleep(3)
                if datetime.now(LOCAL_TZ) - start_poll > retry_timeout:
                    print("[TIMEOUT] Skipping this cycle (no data).")
                    df = None
                    break
                continue

            newest_time = df.index[-1]
            if prev_anchor is None or newest_time > prev_anchor:
                last_candle_time = newest_time
                print(f"[OK] New bar detected: {newest_time.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            else:
                print(f"[WAIT] Latest still {newest_time.strftime('%H:%M')} | retrying...")
                time.sleep(3)

            if datetime.now(LOCAL_TZ) - start_poll > retry_timeout:
                print("[TIMEOUT] No new candle appeared. Skipping.")
                df = None
                break

        if df is None or df.empty:
            continue

        # 3) Indicators & signal
        df = calc_indicators(df)
        if len(df) < max(EMA_SLOW_PERIOD, ATR_PERIOD) + 2:
            print("[WARN] Not enough history for indicators yet.")
            continue

        row_now  = df.iloc[-1]
        row_prev = df.iloc[-2]
        sig = generate_signal(row_now, row_prev)
        print(f"[SIGNAL] {sig or 'NONE'} | close={row_now['close']:.2f} | vwap={row_now['vwap']:.2f} | "
              f"ema9={row_now['ema_fast']:.2f} | ema50={row_now['ema_slow']:.2f} | atr={row_now['atr']:.2f}")

        # 4) Position state (supports multiple)
        positions = get_open_positions(MT5_SYMBOL)
        buy_count, sell_count = count_positions_by_side(positions)
        total_open = len(positions)

        if positions:
            preview = ", ".join([
                ("BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL")
                + f" {fmt_lots(p.volume)} @ {fmt_price(p.price_open)}"
                for p in positions[:5]
            ])
            print(f"[POS] Open={total_open} | BUY={buy_count}, SELL={sell_count} | "
                  f"{preview}{' ...' if total_open > 5 else ''}")
        else:
            print("[POS] NONE OPEN.")

        # 5) (Optional) dedup per bar to prevent multiple entries on the same completed bar
        current_bar_ts = df.index[-1]
        if not ALLOW_MULTIPLE_PER_BAR and sig in ("BUY", "SELL"):
            if last_signal_bar == current_bar_ts:
                print("[DEDUP] Already acted on this bar; skipping new entry.")
                sig = None

        # 6) Trade logic (scalping): multiple entries allowed, with caps
        if sig in ("BUY", "SELL"):
            atr   = float(row_now["atr"])
            price = float(row_now["close"])
            info  = mt5.symbol_info(MT5_SYMBOL)

            if pd.isna(atr) or atr <= 0:
                print("[SKIP] ATR not ready.")
            else:
                stop_distance = SL_MULTIPLIER * atr
                if sig == "BUY":
                    sl_price = price - stop_distance
                    tp_price = price + TP_MULTIPLIER * stop_distance
                else:
                    sl_price = price + stop_distance
                    tp_price = price - TP_MULTIPLIER * stop_distance

                # Decide if we are allowed to add another trade for this side
                side_count = buy_count if sig == "BUY" else sell_count
                can_add_side = side_count < MAX_OPEN_TRADES_PER_SIDE
                can_add_total = total_open < MAX_TOTAL_OPEN_TRADES

                # optional: close opposite side on signal (momentum mode)
                if CLOSE_OPPOSITE_ON_SIGNAL:
                    opposite_side_open = (sell_count if sig == "BUY" else buy_count) > 0
                    if opposite_side_open:
                        print("[ACTION] Closing opposite positions due to new signal...]")
                        for p in list(positions):
                            if (sig == "BUY" and p.type == mt5.ORDER_TYPE_SELL) or (sig == "SELL" and p.type == mt5.ORDER_TYPE_BUY):
                                close_position_market(p)
                                time.sleep(0.3)
                        # refresh state
                        positions = get_open_positions(MT5_SYMBOL)
                        buy_count, sell_count = count_positions_by_side(positions)
                        total_open = len(positions)
                        side_count = buy_count if sig == "BUY" else sell_count
                        can_add_side = side_count < MAX_OPEN_TRADES_PER_SIDE
                        can_add_total = total_open < MAX_TOTAL_OPEN_TRADES

                # throttle order spam
                now_ts = time.time()
                if now_ts - last_order_time < MIN_SECONDS_BETWEEN_ENTRIES:
                    print("[THROTTLE] Waiting to respect min spacing between orders.")
                elif not can_add_total or not can_add_side:
                    print(f"[CAP] Reached caps (total={total_open}/{MAX_TOTAL_OPEN_TRADES}, "
                          f"{sig}={side_count}/{MAX_OPEN_TRADES_PER_SIDE}). No new entry.")
                else:
                    lots = choose_lot_size(mt5.account_info().balance, stop_distance, info)
                    ok = place_order(sig, price, sl_price, tp_price, lots)
                    if ok:
                        last_order_time = time.time()
                        last_signal_bar = current_bar_ts  # mark we acted on this bar

        # 7) next loop -> waits for next 5m close

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping EA (keyboard interrupt).")
        break
    except Exception as e:
        print("[EXCEPTION]", type(e).__name__, str(e))
        time.sleep(2)

# Graceful shutdown
mt5.shutdown()
