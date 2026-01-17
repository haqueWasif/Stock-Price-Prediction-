import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import gradio as gr
import pickle

# Load Model
with open('stock_price_prediction_analysis_target_Trend_Next_Day.pkl', 'rb') as f:
    model_1 = pickle.load(f)
    
with open('stock_price_prediction_analysis_target_Trend_Next_3_Days.pkl', 'rb') as f:
    model_3 = pickle.load(f)
    
with open('stock_price_prediction_analysis_target_Trend_Next_7_Days.pkl', 'rb') as f:
    model_7 = pickle.load(f)
    
with open('stock_price_prediction_analysis_target_Trend_Next_14_Days.pkl', 'rb') as f:
    model_14 = pickle.load(f)
    
with open('stock_price_prediction_analysis_target_Trend_Next_30_Days.pkl', 'rb') as f:
    model_30 = pickle.load(f)
    

# Feature Functions
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def compute_atr(df, window=14):
    high_low = df['High'] - df['Low']

    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1)
    atr = tr.max(axis=1).rolling(window=window).mean()

    return atr


def add_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Time Features
    df['Day_Of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day_Of_Month'] = df['Date'].dt.day
    df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df['Is_Weakend'] = df['Date'].dt.weekday >= 5
    df['Week_Of_Year'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter

    # Lagged Price
    df['Close_Lag_1'] = df['Close'].shift(1) # Previous day's closing price
    df['Close_Lag_2'] = df['Close'].shift(7) # 7-day lag
    df['Close_Lag_3'] = df['Close'].shift(14) # 14-day lag

    # Returns
    df['Log_Return_Lag_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Return_Lag_7'] = np.log(df['Close'] / df['Close'].shift(7))
    df['Log_Return_Lag_14'] = np.log(df['Close'] / df['Close'].shift(14))

    df['Percent_Return'] = df['Close'].pct_change()


    # Technical Features
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()


    df['Volatility'] = df['Log_Return_Lag_1'].rolling(window=14).std()
    df['Return_MA_5'] = df['Log_Return_Lag_1'].rolling(window=5).mean()
    df['Return_MA_10'] = df['Log_Return_Lag_1'].rolling(window=10).mean()


    df['VMA_50'] = df['Volume'].rolling(window=50).mean()

        
    df['RSI_14'] = compute_rsi(df['Close'], period=14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['Close'])
    df['ATR_14'] = compute_atr(df, window=14)
    # Seasonal Decomposition
    if len(df) >= 60:
        decomp = seasonal_decompose(df["Close"], model="multiplicative", period=30, extrapolate_trend="freq")
        df["Trend"] = decomp.trend
        df["Seasonality"] = decomp.seasonal
        df["Residual"] = decomp.resid
    else:
        df["Trend"] = np.nan
        df["Seasonality"] = np.nan
        df["Residual"] = np.nan
    
    return df


def get_data_from_csv(file):
    df = pd.read_csv(file.name)
    
    required = {'Date', 'Sentiment', 'Open', 'High', 'Low', 'Close', 'Volume'}
    
    missing = []
    for col in required:
        if col not in df.columns:
            missing.append(col)
    if missing:
        return pd.DataFrame([{'Error' : f'Missing Columns: {missing}'}])
    
    df_new = add_features(df)
    
    last_day_data = df_new.tail(1)
    
    return last_day_data

def predict_next_days(file):
    latest_data = get_data_from_csv(file)
    
    pred_res = []
            
    pred_model_1 = model_1.predict(latest_data)[0]
    prob_model_1 = model_1.predict_proba(latest_data)[0, 1]
    
    pred_res.append({'Day' : 'Next Day', 'Prediction' : pred_model_1, 'Prob(UP)' : prob_model_1})
            
    pred_model_3 = model_3.predict(latest_data)[0]
    prob_model_3 = model_3.predict_proba(latest_data)[0, 1]
    
    pred_res.append({'Day' : 'Next 3 Days', 'Prediction' : pred_model_3, 'Prob(UP)' : prob_model_3})
            
    pred_model_7 = model_7.predict(latest_data)[0]
    prob_model_7 = model_7.predict_proba(latest_data)[0, 1]
    
    pred_res.append({'Day' : 'Next 7 Days', 'Prediction' : pred_model_7, 'Prob(UP)' : prob_model_7})
    
    pred_model_14 = model_14.predict(latest_data)[0]
    prob_model_14 = model_14.predict_proba(latest_data)[0, 1]
    
    pred_res.append({'Day' : 'Next 14 Days', 'Prediction' : pred_model_14, 'Prob(UP)' : prob_model_14})
    
    pred_model_30 = model_30.predict(latest_data)[0]
    prob_model_30 = model_30.predict_proba(latest_data)[0, 1]
    
    pred_res.append({'Day' : 'Next 30 Days', 'Prediction' : pred_model_30, 'Prob(UP)' : prob_model_30})

    return pd.DataFrame(pred_res)

# Interface
app = gr.Interface(
    fn=predict_next_days,
    inputs=gr.File(label="Upload CSV with Date, Sentiment, Open, High, Low, Close, Volume (>= 250 rows recommended)"),
    outputs=gr.Dataframe(label="Predictions"),
    description='Upload historical OHLCV data',
    title="Stock Trend Predictor",
)
    
app.launch(share=True)
    

    