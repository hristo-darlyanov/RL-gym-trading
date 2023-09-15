import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from enum import Enum
from sklearn.preprocessing import MinMaxScaler
import time
import hmac
import hashlib
import requests
import json
from finta import TA
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

model = PPO.load("saved_models/best_model_800000_LSTM4.zip")


class Positions(int, Enum):
    SHORT = 0
    LONG = 1
    HOLD = 2
    TAKE = 3


def getTickerData(ticker, period, interval):
    hist = yf.download(tickers=ticker, period=period, interval=interval)
    df = pd.DataFrame(hist)
    df = df.reset_index()
    return df


def calculate_percentage_increase(final_value, starting_value):
    try:
        return 100 * ((final_value - starting_value) / starting_value)
    except:
        print(final_value, starting_value)


def supertrend_indicator(df, atr_period, multiplier):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    price_diffs = [high - low, high - close.shift(), close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
    hl2 = (high + low) / 2
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)

    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        curr, prev = i, i - 1

        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        else:
            supertrend[curr] = supertrend[prev]

            if (
                supertrend[curr] == True
                and final_lowerband[curr] < final_lowerband[prev]
            ):
                final_lowerband[curr] = final_lowerband[prev]
            if (
                supertrend[curr] == False
                and final_upperband[curr] > final_upperband[prev]
            ):
                final_upperband[curr] = final_upperband[prev]

        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan

    return final_lowerband, final_upperband


def rsi_indicator(df):
    rsi = TA.RSI(df[["open", "high", "low", "close"]], 14)

    signals = []
    for i in range(0, len(rsi)):
        if rsi[i] > 60:  # Default value: 70
            signals.append(Positions.SHORT)
        elif rsi[i] < 40:  # Default value: 30
            signals.append(Positions.LONG)
        else:
            signals.append(Positions.HOLD)

    buy_signal = [True if signals[n] == 1 else False for n in range(0, len(signals))]
    sell_signal = [True if signals[n] == -1 else False for n in range(0, len(signals))]

    return signals, buy_signal, sell_signal


def swing_detection(index, df):
    sh = []
    sl = []
    start = (index * 2) - 1
    for i in range(index - 1):
        sh.append(False)
        sl.append(False)
    for ci, row in df.iterrows():
        swing_high = False
        swing_low = False

        if ci < start:
            continue

        swing_point_high = df["high"][ci - index]
        swing_point_low = df["low"][ci - index]

        for i in range(0, start):
            swing_high = True
            if i < index:
                if df["high"][ci - i] > swing_point_high:
                    swing_high = False
                    break
            if i > index:
                if df["high"][ci - i] >= swing_point_high:
                    swing_high = False
                    break

        for i in range(0, start):
            swing_low = True
            if i < index:
                if df.low[ci - i] < swing_point_low:
                    swing_low = False
                    break
            if i > index:
                if df.low[ci - i] <= swing_point_low:
                    swing_low = False
                    break

        sh.append(swing_high)
        sl.append(swing_low)

    for i in range(index):
        sh.append(False)
        sl.append(False)

    current_sh = 0
    current_sl = 0
    sh_nums = []
    sl_nums = []
    for i, row in df.iterrows():
        if sh[i] == True:
            current_sh = df.high[i]
        if sl[i] == True:
            current_sl = df.low[i]
        sh_nums.append(current_sh)
        sl_nums.append(current_sl)
    return sh, sl, sh_nums, sl_nums


def money_flow_index_indicator(df, period=14):
    # Calculate typical price (TP) for each period
    df["TP"] = (df["high"] + df["low"] + df["close"]) / 3

    # Calculate raw money flow (RMF) for each period
    df["RMF"] = df["TP"] * df["volume"]

    # Calculate positive and negative money flow
    df["PMF"] = 0.0
    df["NMF"] = 0.0

    for i in range(1, len(df)):
        if df.at[i, "TP"] > df.at[i - 1, "TP"]:
            df.at[i, "PMF"] = df.at[i, "TP"] * df.at[i, "volume"]
        elif df.at[i, "TP"] < df.at[i - 1, "TP"]:
            df.at[i, "NMF"] = df.at[i, "TP"] * df.at[i, "volume"]

    # Calculate money flow ratio (MFR)
    df["MFR"] = (
        df["PMF"].rolling(window=period).sum() / df["NMF"].rolling(window=period).sum()
    )

    # Calculate Money Flow Index (MFI)
    df["MFI"] = 100 - (100 / (1 + df["MFR"]))

    # Remove temporary columns
    df.drop(["TP", "RMF", "PMF", "NMF", "MFR"], axis=1, inplace=True)

    return df["MFI"].values


def produce_prediction(df, window):
    prediction = df.shift(window)["close"] <= df["close"]

    return prediction.astype(int)


def preprocess_data(df):
    scaler = MinMaxScaler()

    mfi_indicator = money_flow_index_indicator(df)
    close_b = produce_prediction(df, 1)
    rsi_signals, _, _ = rsi_indicator(df)
    sh, sl, sh_nums, sl_nums = swing_detection(5, df)
    final_lowerband, final_upperband = supertrend_indicator(df, 10, 3)
    fu_modified = [
        Positions.LONG if not np.isnan(lowerband) else Positions.SHORT
        for lowerband in final_lowerband
    ]

    df["close_binary"] = close_b
    df["mfi"] = mfi_indicator
    df["sh_nums"] = sh_nums
    df["sl_nums"] = sl_nums
    df["supertrend"] = fu_modified
    df["rsi_signals"] = rsi_signals

    df = df.drop(columns={"volume", "Adj Close", "open", "high", "low", "date", "MFI"})
    df = df.dropna()

    df[["mfi"]] = scaler.fit_transform(df[["mfi"]])

    return df


def lambda_handler(event, context):
    df = getTickerData("btc-usd", "max", "1d")
    df = df.rename(
        columns={
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Datetime": "date",
            "Date": "date",
        }
    )
    df = preprocess_data(df)
    df = df[0 : df.shape[0] - 1]

    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # getting prediction for current market data
    prediction = model.predict(
        df[["mfi", "close_binary", "supertrend", "rsi_signals"]].tail(89)
    )[0]
    position_to_take = "hold"
    if prediction == Positions.LONG:
        position_to_take = "BUY"
    elif prediction == Positions.SHORT:
        position_to_take = "SELL"
    print("position_to_take:", position_to_take)

    # Checking if it's valid to trade
    btc_last_price = df["close"].iloc[-1]
    swing_low = df["sl_nums"].iloc[-1]
    swing_high = df["sh_nums"].iloc[-1]
    is_valid_to_trade = False
    if position_to_take != "hold":
        is_valid_to_trade = True
    if position_to_take == "BUY":
        if btc_last_price > swing_low:
            is_valid_to_trade = True
    if position_to_take == "SELL":
        if btc_last_price < swing_high:
            is_valid_to_trade = True

    print("Is it valid to trade: ", is_valid_to_trade)
    HOST_URL = "https://testnet.binancefuture.com"
    END_POINT = "/fapi/v1/order"

    # getting info from database
    docs = (
        db.collection("agents")
        .where(filter=FieldFilter("position", "==", "hold"))
        .stream()
    )
    agents = []

    for doc in docs:
        document = doc.to_dict()
        document["ref"] = doc.id
        agents.append(document)

    if is_valid_to_trade:
        for i, agent in enumerate(agents):
            # Setting up request info
            api_key = agent["apiKey"]
            api_secret = agent["apiSecret"]
            amount_of_usd_to_use = agent["usdtToUse"] * (
                agent["percentagePerTrade"] * 0.01
            )
            converted_quantity = amount_of_usd_to_use / btc_last_price
            converted_quantity = round(float(converted_quantity), 3)
            data_query = f"symbol=BTCUSDT&side={position_to_take}&type=MARKET&quantity={converted_quantity}&timestamp={str(time.time()).replace('.','')[0: 13]}"
            signed_hash_key = hmac.new(
                bytes(api_secret, "latin-1"),
                msg=bytes(data_query, "latin-1"),
                digestmod=hashlib.sha256,
            ).hexdigest()

            data_query = data_query + f"&signature={signed_hash_key}"
            url = HOST_URL + END_POINT + "?" + data_query

            # Sending request
            header = {"X-MBX-APIKEY": api_key, "signature": signed_hash_key}
            response = requests.post(url=url, headers=header)
            print(response.text)

            # Setting up document for database
            reference_to_document = agent.pop("ref")
            agent["held_quantity"] = converted_quantity
            agent["price_of_asset"] = btc_last_price
            agent["position"] = position_to_take

            # Editing document in database
            db.collection("agents").document(reference_to_document).set(agent)

    # getting info from database
    docs = (
        db.collection("agents")
        .where(filter=FieldFilter("position", "!=", "hold"))
        .stream()
    )
    agents = []

    for doc in docs:
        document = doc.to_dict()
        document["ref"] = doc.id
        agents.append(document)

    for i, agent in enumerate(agents):
        trade_done = False
        inverted_position = None

        if agent["position"] == "BUY":
            inverted_position = "SELL"
            if agent["price_of_asset"] < swing_low:
                trade_done = True
        if agent["position"] == "SELL":
            inverted_position = "BUY"
            if agent["price_of_asset"] > swing_high:
                trade_done = True

        if trade_done:
            # Setting up request info
            api_key = agent["apiKey"]
            api_secret = agent["apiSecret"]
            amount_of_usd_to_use = agent["usdtToUse"] * (
                agent["percentagePerTrade"] * 0.01
            )
            converted_quantity = amount_of_usd_to_use / agent["price_of_asset"]
            converted_quantity = round(float(converted_quantity), 3)
            data_query = f"symbol=BTCUSDT&side={inverted_position}&type=MARKET&quantity={converted_quantity}&timestamp={str(time.time()).replace('.','')[0: 13]}"
            signed_hash_key = hmac.new(
                bytes(api_secret, "latin-1"),
                msg=bytes(data_query, "latin-1"),
                digestmod=hashlib.sha256,
            ).hexdigest()

            data_query = data_query + f"&signature={signed_hash_key}"
            url = HOST_URL + END_POINT + "?" + data_query

            # Sending request
            header = {"X-MBX-APIKEY": api_key, "signature": signed_hash_key}
            response = requests.post(url=url, headers=header)
            print(response.text)

            # Setting up document for database
            reference_to_document = agent.pop("ref")
            agent.pop("held_quantity")
            agent.pop("price_of_asset")
            agent["position"] = "hold"

            # Editing document in database
            db.collection("agents").document(reference_to_document).set(agent)

            return {"statusCode": 200, "body": json.dumps("Hello world!")}
