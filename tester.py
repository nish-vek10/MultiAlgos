
"""
Mini tester: OANDA pull + HA/Chandelier-direction signals (M1) for
XAU_USD, BTC_USD, EU50_EUR, USD_JPY. Prints last 20 completed candles and signals.

No MT5 connection. Pure API + terminal debug.

Usage:
  - Set environment variable OANDA_TOKEN to your practice token, or fill the token placeholder below.
  - python tester_CE_M1_debug.py
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime
from pytz import timezone

# ===== CONFIG =====
SYMBOLS = [
    {"name": "XAU_USD"},
    {"name": "BTC_USD"},
    {"name": "EU50_EUR"},
    {"name": "USD_JPY"},
]
GRANULARITY = "M5"
COUNT = 50          # fetch a bit more, then keep the last 20 completed
SHOW_LAST = 20
LOCAL_TZ = timezone("Europe/London")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
# Token: prefer env var, fallback to placeholder
OANDA_TOKEN = os.getenv("OANDA_TOKEN", "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2")

# Keep these aligned with main EA (as requested, unchanged)
ATR_PERIOD = 1
ATR_MULT = 1.85

# ===== UTILS =====
def fetch_oanda_candles(instrument: str, granularity: str = GRANULARITY, count: int = COUNT) -> pd.DataFrame:
    """
    Pull mid-price candles from OANDA. Returns a DataFrame indexed by localized time.
    Keeps only 'complete' candles.
    """
    url = f"{OANDA_API_URL}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"granularity": granularity, "count": count, "price": "M"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for {instrument}: {e}")
        return pd.DataFrame()

    if r.status_code != 200:
        print(f"[ERROR] OANDA {instrument} HTTP {r.status_code}: {r.text}")
        return pd.DataFrame()

    raw = r.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}

    for c in raw:
        if not c.get("complete", False):
            continue
        # parse time as UTC then convert to local
        utc_time = pd.to_datetime(c["time"], utc=True)
        local_time = utc_time.tz_convert(LOCAL_TZ)
        data["time"].append(local_time)
        mid = c["mid"]
        data["open"].append(float(mid["o"]))
        data["high"].append(float(mid["h"]))
        data["low"].append(float(mid["l"]))
        data["close"].append(float(mid["c"]))
        data["volume"].append(int(c.get("volume", 0)))

    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index("time", inplace=True)
    return df


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates your HA calc."""
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["ha_close"].iloc[i - 1]) / 2)

    ha["ha_open"] = ha_open
    ha["ha_high"] = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"]  = pd.concat([df["low"],  ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your original (tester) indicators:
    - TR/ATR from HA prices with ATR_PERIOD (default 1), ATR_MULT (1.85)
    - Chandelier-style stops for direction only
    - BUY/SELL flip detection
    """
    ha = calculate_heikin_ashi(df)
    tr = pd.DataFrame(index=ha.index)
    tr["ha_h"], tr["ha_l"], tr["ha_c"] = ha["ha_high"], ha["ha_low"], ha["ha_close"]
    tr["prev_ha_c"] = tr["ha_c"].shift(1)

    # include for debugging view
    tr["ha_open"], tr["ha_high"], tr["ha_low"] = ha["ha_open"], ha["ha_high"], ha["ha_low"]

    # ATR from HA (as in your EA)
    tr["true_range"] = tr[["ha_h", "ha_l", "prev_ha_c"]].apply(
        lambda row: max(row["ha_h"] - row["ha_l"],
                        abs(row["ha_h"] - row["prev_ha_c"]),
                        abs(row["ha_l"] - row["prev_ha_c"])), axis=1
    )
    tr["atr"] = tr["true_range"].ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    # Chandelier stops (window = ATR_PERIOD)
    long_stop  = tr["ha_h"].rolling(window=ATR_PERIOD).max() - (tr["atr"] * ATR_MULT)
    short_stop = tr["ha_l"].rolling(window=ATR_PERIOD).min() + (tr["atr"] * ATR_MULT)

    long_s, short_s = long_stop.copy(), short_stop.copy()
    for i in range(1, len(tr)):
        if tr["ha_c"].iloc[i-1] > long_s.iloc[i-1]:
            long_s.iloc[i] = max(long_stop.iloc[i], long_s.iloc[i-1])
        else:
            long_s.iloc[i] = long_stop.iloc[i]
        if tr["ha_c"].iloc[i-1] < short_s.iloc[i-1]:
            short_s.iloc[i] = min(short_stop.iloc[i], short_s.iloc[i-1])
        else:
            short_s.iloc[i] = short_stop.iloc[i]

    # Direction & signals
    direction = [1]
    for i in range(1, len(tr)):
        if tr["ha_c"].iloc[i] > short_s.iloc[i-1]:
            direction.append(1)
        elif tr["ha_c"].iloc[i] < long_s.iloc[i-1]:
            direction.append(-1)
        else:
            direction.append(direction[-1])

    tr["dir"] = direction
    tr["dir_prev"] = tr["dir"].shift(1)
    tr["buy_signal"]  = (tr["dir"] == 1)  & (tr["dir_prev"] == -1)
    tr["sell_signal"] = (tr["dir"] == -1) & (tr["dir_prev"] == 1)
    return tr


def continuity_check(idx: pd.DatetimeIndex, expected="min"):
    """
    Basic time continuity check for the last N candles.
    Returns (ok: bool, message: str)
    """
    if len(idx) < 2:
        return False, "Not enough candles for continuity check."
    diffs = pd.Series(idx).diff().dropna()
    one_min = pd.Timedelta(minutes=1)  # or: pd.to_timedelta(1, unit="min")
    share_ok = (diffs == one_min).mean()
    if share_ok > 0.9:
        return True, f"Continuity OK ({share_ok*100:.1f}% 1-min steps)"
    return False, f"Continuity weak ({share_ok*100:.1f}% 1-min steps, possible gaps/off-hours)"


def main():
    if not OANDA_TOKEN or OANDA_TOKEN.startswith("REPLACE_"):
        print("[WARN] OANDA_TOKEN not set. Set env var OANDA_TOKEN for live test calls.")
        # continue anyway; calls will fail if token invalid

    for s in SYMBOLS:
        inst = s["name"]
        print("\n" + "="*70)
        print(f"[FETCH] {inst} | granularity={GRANULARITY} | requesting {COUNT} | keeping last {SHOW_LAST} completed")
        df = fetch_oanda_candles(inst, GRANULARITY, COUNT)

        if df.empty:
            print(f"[ERROR] No data pulled for {inst}.")
            continue

        # Keep last SHOW_LAST completed bars
        df = df.tail(SHOW_LAST).copy()

        # Quick integrity checks
        ok_cont, msg_cont = continuity_check(df.index)
        has_nans = df.isna().any().any()

        print(f"[OK] Received {len(df)} completed candles.")
        print(f"[CHECK] {msg_cont}")
        print(f"[CHECK] NaNs present: {has_nans}")

        # Print raw candlesf
        printable = df.copy()
        printable.index = printable.index.strftime("%Y-%m-%d %H:%M")
        print("\n-- RAW OANDA MID CANDLES (last 20) --")
        print(printable[["open","high","low","close","volume"]])

        # Indicators & signals
        tr = calculate_indicators(df)
        dbg = tr[["ha_c","ha_open","ha_high","ha_low","dir","buy_signal","sell_signal"]].copy()
        dbg.index = dbg.index.strftime("%Y-%m-%d %H:%M")
        dbg["signal"] = dbg.apply(lambda r: "BUY" if r["buy_signal"] else ("SELL" if r["sell_signal"] else ""), axis=1)

        print("\n-- HEIKIN-ASHI + SIGNALS (last 20) --")
        print(dbg[["ha_c","ha_open","ha_high","ha_low","dir","signal"]])

        # Summarize the latest
        latest = tr.iloc[-1]
        latest_signal = "BUY" if latest["buy_signal"] else ("SELL" if latest["sell_signal"] else "NONE")
        last_time = df.index[-1].strftime("%Y-%m-%d %H:%M %Z")
        print(f"\n[SUMMARY] {inst} @ {last_time} | direction={int(latest['dir'])} | signal={latest_signal}")

    print("\n" + "="*70)
    print("[DONE] Tester finished. If all four symbols show candles + signals table, data fetch is good.")

if __name__ == "__main__":
    main()
