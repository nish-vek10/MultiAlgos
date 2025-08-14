"""
Chandelier Exit Strategy for XAUUSD & BTCUSD
Using Heikin-Ashi Candles (1H timeframe)
Fixed SL: 40 (XAU) / 4000 (BTC)
Lot size: 0.1 lots
"""

# ===== FILENAME: CEStrategy_XAU+BTC_1H =====

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
    {"mt5": "XAUUSD", "oanda": "XAU_USD", "fixed_sl": 40},
    {"mt5": "BTCUSD", "oanda": "BTC_USD", "fixed_sl": 4000}
]

timeframe = mt5.TIMEFRAME_H1
num_candles = 500
lot_size = 0.1
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

# Loop through all symbols to print info
for s in symbols:
    symbol_info = mt5.symbol_info(s["mt5"])
    if symbol_info:
        print(f"[INFO] {s['mt5']} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
        print(symbol_info._asdict())
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

def send_order(symbol, action_type, lot, sl_distance):
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
                    send_order(mt5_symbol, mt5.ORDER_TYPE_BUY, lot_size, s['fixed_sl'])

            elif signal == 'SELL':
                if open_position == 'BUY':
                    close_position(position, mt5_symbol)
                if open_position != 'SELL':
                    send_order(mt5_symbol, mt5.ORDER_TYPE_SELL, lot_size, s['fixed_sl'])
        else:
            print(f"[INFO] No new trade action needed for {mt5_symbol}. Signal unchanged: {signal}")

    print("\n[✓] Cycle complete. Waiting for next 1-hour candle...\n")