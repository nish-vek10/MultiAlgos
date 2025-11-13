"""
Crude Oil 16:00 Long Strategy — MT5 Live/Demo Bot
-------------------------------------------------
Logic (Europe/London time):

- Every trading day at exactly 16:00:00:
    • Go LONG on crude oil (MT5 symbol).
    • SL = entry_price - 0.10
    • TP = entry_price + 0.20
    • Risk = 0.50% of current account equity.

- Time exit:
    • If still open at 18:00:00, close at market (time-based exit).

- Weekday filter:
    • Can skip selected weekdays, e.g. skip Wednesdays.

Notes:
- Uses MT5 prices only (no OANDA needed for this live bot).
- 1 trade max per day (per symbol/magic).
"""

import sys
import time
from datetime import datetime, timedelta
from pytz import timezone
import MetaTrader5 as mt5

# =========================
# USER CONFIG
# =========================

# --- MT5 Account / Terminal ---
MT5_LOGIN         = 52603261
MT5_PASSWORD      = "$dqrTIe!5HN7it"
MT5_SERVER        = "ICMarketsSC-Demo"
MT5_TERMINAL_PATH = r"C:\MT5\52603261\terminal64.exe"

# --- Timezone ---
LOCAL_TZ = timezone("Europe/London")

# --- Symbol & Magic ---
MT5_SYMBOL = "XTIUSD"  # <--- CHANGE to your broker's crude symbol (e.g. "WTICOUSD", "USOIL", etc.)
MAGIC      = 770016      # unique ID for this strategy

# --- Strategy parameters ---
RISK_PERCENT = 0.01     # 0.50% of equity per trade
SL_DISTANCE  = 0.10      # price distance (entry - 0.10)
TP_DISTANCE  = 0.20      # price distance (entry + 0.20)

# --- Trading session (local time) ---
ENTRY_TIME_HHMM = "16:00"   # when to enter
EXIT_TIME_HHMM  = "19:00"   # latest time to be in trade (time-exit)

# --- Weekday filter (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun) ---
# Example:
#   {2}          = skip Wednesdays only
#   {5, 6}       = skip weekends
#   set()        = trade every day
SKIP_WEEKDAYS = set()

# --- Loop behaviour ---
SLEEP_SECONDS = 0.5   # main loop sleep interval
ENTRY_TOLERANCE_SECONDS = 5  # only enter within first 5s after 16:00:00


# =========================
# MT5 INITIALISATION
# =========================

def init_mt5():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD,
                          server=MT5_SERVER, path=MT5_TERMINAL_PATH):
        print("[ERROR] MT5 initialization failed:", mt5.last_error())
        sys.exit(1)

    acct = mt5.account_info()
    if acct is None:
        print("[ERROR] Failed to retrieve MT5 account info:", mt5.last_error())
        sys.exit(1)

    print(f"[MT5] Connected: login={acct.login}, "
          f"balance={acct.balance:.2f}, equity={acct.equity:.2f}")

    info = mt5.symbol_info(MT5_SYMBOL)
    if info is None:
        print(f"[ERROR] Symbol not found: {MT5_SYMBOL}")
        sys.exit(1)
    if not info.visible:
        if not mt5.symbol_select(MT5_SYMBOL, True):
            print(f"[ERROR] Failed to select symbol: {MT5_SYMBOL}")
            sys.exit(1)

    print(f"[MT5] Symbol {MT5_SYMBOL}: digits={info.digits}, "
          f"contract_size={info.trade_contract_size}, "
          f"vol_min={info.volume_min}, vol_max={info.volume_max}, step={info.volume_step}")

    return info


# =========================
# HELPERS
# =========================

SYMBOL_INFO = None  # will be filled by init_mt5()


def now_local():
    return datetime.now(LOCAL_TZ)


def get_today_entry_exit_times():
    """Return today's entry and exit datetimes in LOCAL_TZ."""
    n = now_local()
    h_e, m_e = map(int, ENTRY_TIME_HHMM.split(":"))
    h_x, m_x = map(int, EXIT_TIME_HHMM.split(":"))

    entry_dt = n.replace(hour=h_e, minute=m_e, second=0, microsecond=0)
    exit_dt  = n.replace(hour=h_x, minute=m_x, second=0, microsecond=0)

    # Handle midnight rollover (not needed here, but safe)
    if exit_dt <= entry_dt:
        exit_dt += timedelta(days=1)

    return entry_dt, exit_dt


def round_price(x: float) -> float:
    d = SYMBOL_INFO.digits
    return round(float(x), d)


def round_lots(lots: float) -> float:
    """Respect volume_min, volume_max, volume_step."""
    step = max(SYMBOL_INFO.volume_step, 0.01)
    vol_min = SYMBOL_INFO.volume_min
    vol_max = SYMBOL_INFO.volume_max
    # round to nearest step
    rounded = round(round(lots / step) * step, 2)
    return max(vol_min, min(rounded, vol_max))


def enforce_stop_level(order_type, price, sl_price, tp_price):
    """
    Ensure SL/TP respect broker's minimum stop distance.
    """
    stop_level_points = getattr(SYMBOL_INFO, "trade_stops_level", 0)
    point = SYMBOL_INFO.point
    if stop_level_points and stop_level_points > 0:
        min_dist = stop_level_points * point
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
    return round_price(sl_price), round_price(tp_price)


def calc_lot_size(stop_distance_price: float) -> float:
    """
    Compute lot size so that:
        risk_dollars = equity * RISK_PERCENT
        risk_dollars ≈ stop_distance_price * contract_size * lots
    """
    acct = mt5.account_info()
    if acct is None:
        print("[ERROR] account_info() is None")
        return 0.0

    equity = acct.equity
    risk_dollars = equity * RISK_PERCENT

    contract_size = SYMBOL_INFO.trade_contract_size
    if stop_distance_price <= 0 or contract_size <= 0:
        return round_lots(SYMBOL_INFO.volume_min)

    raw_lots = risk_dollars / (stop_distance_price * contract_size)
    return round_lots(raw_lots)


def get_our_positions():
    poss = mt5.positions_get(symbol=MT5_SYMBOL)
    if not poss:
        return []
    return [p for p in poss if p.magic == MAGIC]


def place_entry_long():
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None:
        print("[ERROR] No tick data available for", MT5_SYMBOL)
        return False

    price = round_price(tick.ask)
    sl_price = price - SL_DISTANCE
    tp_price = price + TP_DISTANCE

    sl_price, tp_price = enforce_stop_level(mt5.ORDER_TYPE_BUY, price, sl_price, tp_price)

    lots = calc_lot_size(SL_DISTANCE)
    if lots <= 0:
        print("[ERROR] Lot size computed as <= 0, aborting entry.")
        return False

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": MAGIC,
        "comment": "Crude_16pm_Long",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    res = mt5.order_send(req)
    if res is None:
        print("[ERROR] order_send() returned None, last_error:", mt5.last_error())
        return False
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Entry failed: retcode={res.retcode}, comment={res.comment}")
        return False

    print(f"[OK] BUY {lots:.2f} {MT5_SYMBOL} @ {price:.{SYMBOL_INFO.digits}f} "
          f"SL={sl_price:.{SYMBOL_INFO.digits}f} TP={tp_price:.{SYMBOL_INFO.digits}f}")
    return True


def close_all_our_positions(reason: str = "TimeExit"):
    poss = get_our_positions()
    if not poss:
        return

    print(f"[CLOSE] Closing {len(poss)} position(s) for {MT5_SYMBOL} (reason={reason})...")
    for p in poss:
        if p.type == mt5.ORDER_TYPE_BUY:
            action = mt5.ORDER_TYPE_SELL
        else:
            action = mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(MT5_SYMBOL)
        if tick is None:
            print("[ERROR] No tick to close position", p.ticket)
            continue

        price = tick.bid if action == mt5.ORDER_TYPE_SELL else tick.ask
        price = round_price(price)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": MT5_SYMBOL,
            "volume": p.volume,
            "type": action,
            "position": p.ticket,
            "price": price,
            "deviation": 20,
            "magic": p.magic,
            "comment": f"Crude_16pm_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Close failed: ticket={p.ticket}, "
                  f"retcode={getattr(res, 'retcode', None)}, "
                  f"comment={getattr(res, 'comment', None)}")
        else:
            print(f"[OK] Closed ticket={p.ticket} at {price:.{SYMBOL_INFO.digits}f}")


# =========================
# MAIN LOOP
# =========================

def main_loop():
    print("[RUN] Crude 16:00 Long bot started.")

    last_entry_date = None  # to prevent more than 1 trade per day

    while True:
        try:
            now = now_local()
            today = now.date()
            weekday = now.weekday()

            # Compute today's entry/exit times
            entry_dt, exit_dt = get_today_entry_exit_times()

            # 1) Time-based exit: after exit_dt, ensure no open positions for this magic
            if now >= exit_dt:
                poss = get_our_positions()
                if poss:
                    print(f"[INFO] {now} reached exit time {exit_dt}, closing open positions.")
                    close_all_our_positions("TimeExit")

            # 2) Skip weekdays if configured
            if weekday in SKIP_WEEKDAYS:
                # Reset last_entry_date if we moved to a new day
                if last_entry_date is not None and last_entry_date != today:
                    last_entry_date = None
                time.sleep(SLEEP_SECONDS)
                continue

            # 3) Entry logic: between entry_dt and entry_dt + tolerance, one trade per day
            seconds_since_entry = (now - entry_dt).total_seconds()
            if (0 <= seconds_since_entry <= ENTRY_TOLERANCE_SECONDS
                and (last_entry_date is None or last_entry_date != today)):

                positions = get_our_positions()
                if positions:
                    # Already have a position (maybe from manual/startup), avoid double entry
                    last_entry_date = today
                else:
                    print(f"[ENTRY] {now} ~ attempting BUY at scheduled time {entry_dt}")
                    ok = place_entry_long()
                    if ok:
                        last_entry_date = today

            # 4) New day housekeeping: reset last_entry_date when date changes & no position
            if last_entry_date is not None and last_entry_date != today:
                # fresh day; allow a new trade
                last_entry_date = None

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("\n[EXIT] Stopping bot (KeyboardInterrupt).")
            break
        except Exception as e:
            print("[EXCEPTION]", type(e).__name__, str(e))
            time.sleep(2)


if __name__ == "__main__":
    SYMBOL_INFO = init_mt5()
    try:
        main_loop()
    finally:
        mt5.shutdown()
        print("[MT5] Shutdown complete.")
