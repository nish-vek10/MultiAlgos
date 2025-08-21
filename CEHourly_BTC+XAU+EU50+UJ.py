"""
Chandelier-Direction, Fixed SL/TP (1H) for XAUUSD, BTCUSD, EURO STOXX 50, USDJPY
- Signals: Heikin-Ashi based direction flip (Chandelier-style stops for direction only).
- Orders: Market orders with per-symbol fixed SL and fixed TP.
- Position size: Per-asset lot sizes (configurable) to keep risk roughly aligned across symbols.
  Example: XAU $40 @ 0.1 lots ≈ $400 risk (≈0.4% on a $100k account).

  ~ USDJPY: SL = 2.00 JPY (≈ 200 pips). For 1 lot, pip ≈ $6–7 → risk per lot ≈ $1,200–1,400. Target $400 ⇒ ~0.30 lots.
  ~ STOXX50: SL = 50 pts. If your contract is €1/pt/lot, risk per lot ≈ €50 → target ~$400 ≈ €360 ⇒ ~7.2 lots.
"""

# ===== FILENAME: CEHourly_BTC+XAU+EU50+UJ =====

import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
import os
import sys
import requests
import pandas as pd
from pytz import timezone

# === CONFIGURATION === #
symbols = [
    {"mt5": "XAUUSD", "oanda": "XAU_USD", "fixed_sl": 40, "fixed_tp": 40, "lot_size": 0.1},
    {"mt5": "BTCUSD", "oanda": "BTC_USD", "fixed_sl": 4000, "fixed_tp": 4000, "lot_size": 0.1},
    {"mt5": "STOXX50", "oanda": "EU50_EUR", "fixed_sl": 50, "fixed_tp": 50, "risk_usd": 400.0},
    {"mt5": "USDJPY", "oanda": "USD_JPY", "fixed_sl": 2, "fixed_tp": 2, "risk_usd": 400.0},
]

timeframe = mt5.TIMEFRAME_H1
num_candles = 500
lot_size = 0.1  # fallback only; normally overridden per symbol in 'symbols'
slippage = 10
atr_period = 1
atr_mult = 1.85
magic_number = 987654 # Unique ID for this EA's trades
local_tz = timezone('Europe/London')

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52461477
mt5_password = "F!3HXK2m9U22$c"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\MT5\52461477\terminal64.exe"

print("MT5 Path Exists?", os.path.exists(mt5_terminal_path))

# === CONNECT TO MT5 === #
if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server, path=mt5_terminal_path):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    quit()

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 Account Connected! \nAccount: {account.login} \nBalance: ${account.balance:.2f}\n")

def get_tick_specs(symbol):
    """
    Return (tick_size, tick_value, unit, profit_ccy) for 1.0 lot.
    unit = "account"  -> tick_value already in account currency (from trade_* fields)
    unit = "profit"   -> tick_value in profit currency (from generic fields), needs FX conversion
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return (None, None, None, None)

    # Prefer trade_* (usually in account currency)
    t_size = getattr(info, "trade_tick_size", None)
    t_val  = getattr(info, "trade_tick_value", None)
    if t_size and t_val:
        return (t_size, t_val, "account", getattr(info, "currency_profit", None))

    # Fallback to generic fields (usually in profit currency units)
    t_size = getattr(info, "tick_size", None) or getattr(info, "point", None)
    t_val  = getattr(info, "tick_value", None)
    return (t_size, t_val, "profit", getattr(info, "currency_profit", None))

def convert_to_account_currency(amount, from_ccy, account_ccy="USD"):
    """
    Convert amount from 'from_ccy' to 'account_ccy' using MT5 quotes (mid).
    Tries direct pair (FROMCCYACCOUNTCCY), inverse, then bridges via USD.
    Returns None if conversion unavailable.
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

    # Try direct (e.g., EURUSD)
    direct = f"{from_ccy}{account_ccy}"
    m = mid(direct)
    if m:
        return amount * m

    # Try inverse (e.g., USDEUR)
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
    Compute lots so that risk (in account currency) ≈ target_usd, using:
      risk(1 lot) = (SL / tick_size) * tick_value [profit ccy]
      -> convert to account_ccy, then lots = target_usd / risk(1 lot in account_ccy)
    Enforces broker lot step/min/max and returns a float or None.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    tick_size, tick_value, unit, profit_ccy = get_tick_specs(symbol)
    if not tick_size or not tick_value or tick_size == 0:
        return None

    ticks = sl_distance / tick_size
    risk_per_lot = ticks * float(tick_value)  # risk per 1.0 lot in 'unit' currency

    # Convert only if tick_value is in profit currency
    if unit == "profit":
        risk_per_lot_acct = convert_to_account_currency(risk_per_lot, profit_ccy, account_ccy)
    else:  # 'account'
        risk_per_lot_acct = risk_per_lot

    if not risk_per_lot_acct or risk_per_lot_acct <= 0:
        return None

    raw_lots = float(target_usd) / risk_per_lot_acct

    # Quantize to broker constraints
    step = info.volume_step or 0.01
    raw_lots = round(raw_lots / step) * step
    raw_lots = max(raw_lots, info.volume_min)
    raw_lots = min(raw_lots, info.volume_max)

    # tidy formatting to step precision
    step_str = f"{step:.10f}".rstrip("0")
    dec = len(step_str.split(".")[1]) if "." in step_str else 0
    return float(f"{raw_lots:.{dec}f}")

def estimate_risk(symbol, sl_distance, lots, account_ccy="USD"):
    """
    Estimate risk for given lots in account currency AND in profit currency (for transparency).
    Returns (risk_in_account_ccy, profit_ccy, risk_in_profit_ccy) or (None, None, None) on failure.
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


# Loop through all symbols to print info
for s in symbols:
    symbol_info = mt5.symbol_info(s["mt5"])
    if symbol_info:
        print(f"[INFO] {s['mt5']} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
        print(symbol_info._asdict())

        eff_lot = s.get("lot_size", lot_size)  # fallback lot
        target_usd = s.get("risk_usd", None)

        # If risk_usd specified, compute dynamic lot for that target
        if target_usd is not None:
            dyn_lot = lots_for_target_usd(s["mt5"], s["fixed_sl"], target_usd, account.currency)
            if dyn_lot is not None:
                eff_lot = dyn_lot
                print(f"[SIZE] {s['mt5']}: risk_usd={target_usd} -> computed lot ≈ {eff_lot}")
            else:
                print(f"[SIZE] {s['mt5']}: risk_usd={target_usd} -> unable to compute dynamic lot, using fallback lot={eff_lot}")

        # Risk estimates (both in account ccy and profit ccy for transparency)
        r_acct, p_ccy, r_profit = estimate_risk(s["mt5"], s["fixed_sl"], eff_lot, account.currency)
        if r_acct is not None:
            extra = f" (≈ {r_profit:.2f} {p_ccy})" if r_profit is not None else ""
            print(
                f"[RISK] {s['mt5']}: lot={eff_lot} | SL={s['fixed_sl']} | TP={s['fixed_tp']} -> est ≈ {r_acct:.2f} {account.currency}{extra}")
        else:
            print(
                f"[RISK] {s['mt5']}: lot={eff_lot} | SL={s['fixed_sl']} | TP={s['fixed_tp']} -> est risk unavailable (tick specs/conversion missing)")
    else:
        print(f"[ERROR] Unable to fetch symbol info for {s['mt5']}")


oanda_token = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_account_id = "101-004-35770497-001"
oanda_api_url = "https://api-fxpractice.oanda.com/v3"


def fetch_oanda_candles(symbol, granularity="H1", count=500):
    """
    Fetch candle data from OANDA REST API.
    symbol: OANDA instrument name "XAU_"
    granularity: M1, M5, M15, H1, D etc.
    count: number of candles to fetch
    """
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch candles from OANDA {symbol}:", response.status_code, response.text)
        return None

    raw_candles = response.json().get("candles", [])
    data = {
        "time": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": []
    }

    for candle in raw_candles:
        if candle['complete']:
            utc_time = pd.to_datetime(candle['time'])
            local_time = utc_time.tz_convert(local_tz)
            data['time'].append(local_time)
            data['open'].append(float(candle['mid']['o']))
            data['high'].append(float(candle['mid']['h']))
            data['low'].append(float(candle['mid']['l']))
            data['close'].append(float(candle['mid']['c']))
            data['volume'].append(int(candle['volume']))

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df

def get_position(symbol):
    """
    Check for existing position on a symbol.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if p.magic == magic_number:
            return p
    return None

def send_order(symbol, action_type, lot, sl_distance, tp_distance=None):
    """
    Send a buy or sell order.
    """
    # === VOLUME VALIDATION ===
    if lot is None or lot <= 0:
        print(f"[ERROR] Invalid trade volume: {lot}")
        return

    # === SYMBOL INFO CHECK ===
    # Ensure symbol is selected
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

    # === LOT SIZE VALIDATION ===
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step

    if lot < symbol_info.volume_min or lot > symbol_info.volume_max:
        print(f"[ERROR] Lot size {lot} out of range: min={symbol_info.volume_min}, max={symbol_info.volume_max}")
        return

    # === TICK DATA VALIDATION ===
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return

    price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid
    sl_price = price - sl_distance if action_type == mt5.ORDER_TYPE_BUY else price + sl_distance

    tp_price = None
    if tp_distance is not None:
        tp_price = price + tp_distance if action_type == mt5.ORDER_TYPE_BUY else price - tp_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": price,
        "sl": sl_price,
        "deviation": slippage,
        "magic": magic_number,
        "comment": "ChandelierEntryBot",
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
        print(f"[OK] Order placed successfully on {symbol}: Ticket={result.order}")

def close_position(position, symbol):
    """
    Close an open position.
    """
    if position is None:
        return
    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
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
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {result.retcode}, {result.comment}")
    else:
        print(f"[OK] Position closed: {result}")

# === HEIKIN ASHI CALCULATION === #
def calculate_heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)

    ha_df['ha_open'] = ha_open

    ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
    ha_df['ha_low'] = pd.concat([df['low'], ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)
    return ha_df

# === CALCULATE INDICATORS: ATR, Stops, Direction === #
def calculate_indicators(df):
    ha_df = calculate_heikin_ashi(df)

    tr = pd.DataFrame(index=ha_df.index)
    tr['ha_h'] = ha_df['ha_high']
    tr['ha_l'] = ha_df['ha_low']
    tr['ha_c'] = ha_df['ha_close']
    tr['prev_ha_c'] = tr['ha_c'].shift(1)

    # Include for debugging/visualisation
    tr['ha_open'] = ha_df['ha_open']
    tr['ha_high'] = ha_df['ha_high']
    tr['ha_low'] = ha_df['ha_low']

    # True Range & ATR
    tr['true_range'] = tr[['ha_h', 'ha_l', 'prev_ha_c']].apply(
        lambda row: max(
            row['ha_h'] - row['ha_l'],
            abs(row['ha_h'] - row['prev_ha_c']),
            abs(row['ha_l'] - row['prev_ha_c'])
        ), axis=1
    )

    tr['atr'] = tr['true_range'].ewm(alpha=1/atr_period, adjust=False).mean()

    # Chandelier Stops
    long_stop = tr['ha_h'].rolling(window=atr_period).max() - (tr['atr'] * atr_mult)
    short_stop = tr['ha_l'].rolling(window=atr_period).min() + (tr['atr'] * atr_mult)

    long_stop_smooth = long_stop.copy()
    short_stop_smooth = short_stop.copy()

    for i in range(1, len(tr)):
        if tr['ha_c'].iloc[i - 1] > long_stop_smooth.iloc[i - 1]:
            long_stop_smooth.iloc[i] = max(long_stop.iloc[i], long_stop_smooth.iloc[i - 1])
        else:
            long_stop_smooth.iloc[i] = long_stop.iloc[i]
        if tr['ha_c'].iloc[i - 1] < short_stop_smooth.iloc[i - 1]:
            short_stop_smooth.iloc[i] = min(short_stop.iloc[i], short_stop_smooth.iloc[i - 1])
        else:
            short_stop_smooth.iloc[i] = short_stop.iloc[i]

    # Direction Detection
    dir = [1]
    for i in range(1, len(tr)):
        if tr['ha_c'].iloc[i] > short_stop_smooth.iloc[i - 1]:
            dir.append(1)
        elif tr['ha_c'].iloc[i] < long_stop_smooth.iloc[i - 1]:
            dir.append(-1)
        else:
            dir.append(dir[-1])

    tr['dir'] = dir
    tr['dir_prev'] = tr['dir'].shift(1)
    tr['buy_signal'] = (tr['dir'] == 1) & (tr['dir_prev'] == -1)
    tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] == 1)

    return tr

# === MAIN LOOP === #
last_candle_time = {s['mt5']: None for s in symbols}
last_signal = {s['mt5']: None for s in symbols}
last_signal_info = {s['mt5']: None for s in symbols}  # Stores time and signal info

while True:
    # WAIT UNTIL NEXT 1-HOUR CANDLE
    now = datetime.now()
    seconds = now.minute * 60 + now.second
    sleep_seconds = (3600 - (seconds % 3600)) % 3600

    # seconds = now.minute * 60 + now.second
    # sleep_seconds = (3600 - (seconds % 3600)) % 3600

    print("\n[*] Waiting until next 1-hour candle close...\n")
    while sleep_seconds > 0:
        wait_time_formatted = str(timedelta(seconds=sleep_seconds))
        # sys.stdout.write(f"\rTime remaining: {wait_time_formatted} ")   # timer countdown
        # sys.stdout.flush()
        time.sleep(1)
        sleep_seconds -= 1

    print("\n[+] 1-hour candle closed. Fetching data...\n")

    # PROCESS EACH SYMBOL
    for s in symbols:
        oanda_symbol = s['oanda']
        mt5_symbol = s['mt5']
        print(f"\n==================== {mt5_symbol} ====================")

        retry_timeout = timedelta(minutes=2)
        start_time = datetime.now()
        df = None

        # --- Retry until new candle or timeout ---
        while True:
            df = fetch_oanda_candles(symbol=oanda_symbol, granularity="H1", count=500)

            if df is None or df.empty:
                print(f"[ERROR] Failed to retrieve data for {mt5_symbol}. Retrying in 10s...")
                time.sleep(10)
                continue

            latest_candle_time = df.index[-1]

            if last_candle_time[mt5_symbol] is None or latest_candle_time > last_candle_time[mt5_symbol]:
                last_candle_time[mt5_symbol] = latest_candle_time
                print(f"[OK] New 1-hour candle detected: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            else:
                print(f"[WAIT] No new candle yet for {mt5_symbol}. Latest: {latest_candle_time.strftime('%H:%M:%S')} | Retrying in 10s...")
                time.sleep(10)

            if datetime.now() - start_time > retry_timeout:
                print(f"[TIMEOUT] No new candle for {mt5_symbol} after {retry_timeout.seconds} seconds. Skipping...")
                df = None
                break

        # Skip symbol if no valid data
        if df is None or df.empty:
            continue

        # PRINT RAW CANDLES
        print(f"[OK] Retrieved {len(df)} candles from OANDA for {mt5_symbol}")
        print("\n= = = RAW OANDA CANDLESTICK DATA (LAST 20) = = =")
        print(df.assign(time=df.index.strftime('%Y-%m-%d %H:%M')).set_index('time').tail(20))

        # CALCULATE INDICATORS & SIGNALS
        tr = calculate_indicators(df)
        latest = tr.iloc[-1]

        debug_df = tr[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'buy_signal', 'sell_signal']].copy()
        debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
        debug_df['signal'] = debug_df.apply(
            lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1
        )
        print(f"\n= = = LAST 20 HEIKIN-ASHI CANDLES ({mt5_symbol}) = = =")
        print(debug_df[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'signal']].tail(20))

        # Determine signal
        signal = 'BUY' if latest['buy_signal'] else 'SELL' if latest['sell_signal'] else None

        # Get current open position
        position = get_position(mt5_symbol)
        open_position = 'BUY' if position and position.type == mt5.ORDER_TYPE_BUY else 'SELL' if position else None

        # Always show last executed signal info
        if last_signal_info[mt5_symbol]:
            print(f"[INFO] Last executed signal for {mt5_symbol}: {last_signal_info[mt5_symbol]}")
        else:
            print(f"[INFO] No previous signal executed yet for {mt5_symbol}.")

        # Handle cases with no signal
        if not signal:
            print(f"[INFO] No BUY or SELL signal detected for {mt5_symbol}.")
            continue

        # Only act if the signal changes or position is different
        if signal != last_signal[mt5_symbol] or open_position != signal:
            prev_sig = last_signal[mt5_symbol] if last_signal[mt5_symbol] else "NONE"
            print(f"[TRADE] New signal for {mt5_symbol}: {signal} | Prev: {prev_sig} | Open pos: {open_position or 'NONE'}")

            # Update last signal memory with timestamp
            last_signal[mt5_symbol] = signal
            last_signal_info[mt5_symbol] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {signal}"

            if signal == 'BUY':
                if open_position == 'SELL':
                    close_position(position, mt5_symbol)
                if open_position != 'BUY':
                    # Dynamic sizing if risk_usd present; otherwise fallback to per-symbol lot or global
                    use_lot = s.get('lot_size', lot_size)
                    if 'risk_usd' in s:
                        dyn_lot = lots_for_target_usd(mt5_symbol, s['fixed_sl'], s['risk_usd'], account.currency)
                        if dyn_lot is not None:
                            use_lot = dyn_lot

                    send_order(mt5_symbol, mt5.ORDER_TYPE_BUY, use_lot, s['fixed_sl'], s['fixed_tp'])

            elif signal == 'SELL':
                if open_position == 'BUY':
                    close_position(position, mt5_symbol)
                if open_position != 'SELL':
                    use_lot = s.get('lot_size', lot_size)
                    if 'risk_usd' in s:
                        dyn_lot = lots_for_target_usd(mt5_symbol, s['fixed_sl'], s['risk_usd'], account.currency)
                        if dyn_lot is not None:
                            use_lot = dyn_lot

                    send_order(mt5_symbol, mt5.ORDER_TYPE_SELL, use_lot, s['fixed_sl'], s['fixed_tp'])
        else:
            print(f"[INFO] No new trade action needed for {mt5_symbol}. Signal unchanged: {signal}")

    print("\n[✓] Cycle complete. Waiting for next 1-hour candle...\n")

