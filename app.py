import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from datetime import timedelta , datetime
from flask import Flask, render_template
import joblib 
import matplotlib.pyplot as plt

app = Flask(__name__)

#.\env\Scripts\activate 

# Constants
FILE_PATH = r"C:\Users\sumit\OneDrive\Desktop\kns\uploads\5min_candles.csv" # Path to your CSV file

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.set_index('Date', inplace=True)
    return df

# Load live market data from yFinance and filter for market hours
def load_live_data(ticker, period='1d', interval='5m'):
    data = yf.download(ticker, period=period, interval=interval)

    if not data.empty:
        data.index = data.index.tz_convert('Asia/Kolkata')
        market_open = data.between_time("09:15", "15:30")
        ohlc_data = market_open[['Open', 'High', 'Low', 'Close']].copy()
        ohlc_data = ohlc_data.reset_index()
        ohlc_data.rename(columns={'Datetime': 'Date'}, inplace=True)
        ohlc_data['Date'] = pd.to_datetime(ohlc_data['Date']).dt.tz_localize(None)
        ohlc_data.set_index('Date', inplace=True)
        return ohlc_data
    else:
        print("No live data available.")
        return pd.DataFrame()

# Add technical indicators using pandas_ta
def add_technical_indicators(df):
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Fix the Stochastic indicator unpacking issue
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], length=14)
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']

    return df


# Prepare the dataset by adding patterns and indicators
def prepare_data(df):
    df['Doji'] = np.where((df['Close'] - df['Open']).abs() < (df['High'] - df['Low']) * 0.1, 1, 0)
    df['Hammer'] = np.where((df['Close'] - df['Open']).abs() < (df['High'] - df['Low']) * 0.3, 1, 0)
    df['Engulfing'] = np.where((df['Open'] < df['Close']) & (df['Open'].shift(1) > df['Close'].shift(1)) & (df['Open'] < df['Close'].shift(1)), 1, 0)
    df['Shooting_Star'] = np.where((df['Open'] - df['Close']).abs() < (df['High'] - df['Low']) * 0.3, 1, 0)
    df['Inverted_Hammer'] = np.where((df['Open'] - df['Close']).abs() < (df['High'] - df['Low']) * 0.3, 1, 0)
    df['Morning_Star'] = np.where((df['Close'] < df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(-1) > df['Open'].shift(-1)), 1, 0)
    df['Evening_Star'] = np.where((df['Close'] > df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(-1) < df['Open'].shift(-1)), 1, 0)
    df['Three_Black_Crows'] = np.where((df['Close'] < df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(2) < df['Open'].shift(2)), 1, 0)
    df['Three_White_Soldiers'] = np.where((df['Close'] > df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(2) > df['Open'].shift(2)), 1, 0)

    df['Body_Size'] = (df['Close'] - df['Open']).abs()
    df['Range'] = df['High'] - df['Low']

    features = ['Body_Size', 'Range', 'SMA_20', 'SMA_50', 'EMA_9', 'RSI', 'MACD', 'ADX', 'CCI', 'ATR', 'Stoch_K', 'Stoch_D', 'Doji', 'Hammer', 'Engulfing', 'Shooting_Star', 'Inverted_Hammer', 'Morning_Star', 'Evening_Star', 'Three_Black_Crows', 'Three_White_Soldiers']
    df = df.dropna(subset=features + ['Close'])
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df[:-1]

    X = df[features]
    y = df['Target']

    return X, y

# Train XGBoost model for pattern detection
def train_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save the trained model
    joblib.dump(model, 'pattern_model.pkl')

    return model

# Prepare the dataset for price prediction
def prepare_data_for_price_prediction(df):
    features = ['Body_Size', 'Range', 'SMA_20', 'SMA_50', 'EMA_9', 'RSI', 'MACD', 'ADX', 'CCI', 'ATR', 'Stoch_K', 'Stoch_D', 'Doji', 'Hammer', 'Engulfing', 'Shooting_Star', 'Inverted_Hammer', 'Morning_Star', 'Evening_Star', 'Three_Black_Crows', 'Three_White_Soldiers']
    df['Target_Close'] = df['Close'].shift(-1)
    df = df.dropna(subset=features + ['Target_Close'])


    X = df[features]
    y = df['Target_Close']

    return X, y

# Train XGBoost regression model for price prediction
def train_price_prediction_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    price_model = XGBRegressor(n_estimators=100, random_state=42)
    price_model.fit(X_train, y_train)
    predictions = price_model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R²): {r2}')

    # Save the trained price prediction model
    joblib.dump(price_model, 'price_prediction_model.pkl')

    return price_model

def get_next_working_day(current_date):
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day
#####
def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        print(f"Model file {model_filename} not found.")
        return None

def predict_next_candle_close(df, price_model):
    # Set index as datetime
    df.index = pd.to_datetime(df.index, errors='coerce')

    # Features for prediction
    features = ['Body_Size', 'Range', 'SMA_20', 'SMA_50', 'EMA_9', 'RSI', 'MACD', 'ADX',
                'CCI', 'ATR', 'Stoch_K', 'Stoch_D', 'Doji', 'Hammer', 'Engulfing',
                'Shooting_Star', 'Inverted_Hammer', 'Morning_Star', 'Evening_Star',
                'Three_Black_Crows', 'Three_White_Soldiers']

    # Get latest data for prediction
    latest_data = df.iloc[-1][features].to_frame().T
    predicted_close = price_model.predict(latest_data)

    # Determine the time of the next prediction
    last_candle_time = df.index[-1]
    if pd.isna(last_candle_time):
        print("Error: last_candle_time is NaT")
        return

    # Define the end of the trading day and check if the next prediction is after 3:25 PM
    end_of_day_time = datetime.strptime("15:25", "%H:%M").time()
    if last_candle_time.time() >= end_of_day_time:
        # If it's after 3:25 PM, predict for 9:15 AM on the next working day
        next_day = get_next_working_day(last_candle_time)
        next_candle_time = next_day.replace(hour=9, minute=15)
    else:
        # Otherwise, add 5 minutes for the next intraday candle
        next_candle_time = last_candle_time + timedelta(minutes=5)

    print(f"Predicted Close for {next_candle_time}: {predicted_close[0]}")
    return predicted_close[0], next_candle_time





####
def graph_plots(combined_data, price_model):
    # Recent period for Nov 18 - Nov 22
    recent_period = combined_data.loc['2024-11-18':'2024-11-22']
    predicted_closes = []
    actual_closes = recent_period['Close'].values

    # Predict closes for the recent period
    for i in range(len(recent_period)):
        predicted_close, _ = predict_next_candle_close(recent_period.iloc[:i+1], price_model)
        predicted_closes.append(predicted_close)

    # Plot Actual vs Predicted Close Prices (Nov 18 - Nov 22)
    plt.figure(figsize=(10, 6))
    plt.plot(recent_period.index, actual_closes, label='Actual Close Prices', color='blue')
    plt.plot(recent_period.index, predicted_closes, label='Predicted Close Prices', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Actual vs. Predicted Close Prices (Nov 18 - Nov 22)')
    plt.legend()
 
    # Save the plot as an image (optional)
    plot_filename = 'static/prediction_plot.png'
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename


#####



@app.route('/')
def index():
    return render_template('index.html')


# Flask route to trigger the prediction process
# @app.route('/predict', methods=['GET'])
# def predict():
#     historical_data = load_data_from_csv(FILE_PATH)
#     live_data = load_live_data('^NSEI')

#     # Rename the columns of live_data to match historical_data
#     live_data.columns = ['Open', 'High', 'Low', 'Close']
#     combined_data = pd.concat([historical_data, live_data], axis=0)

#     # Add technical indicators
#     data_with_indicators = add_technical_indicators(combined_data)

#     # Prepare data for pattern recognition and price prediction
#     X, y = prepare_data(data_with_indicators)
#     X_price, y_price = prepare_data(data_with_indicators)

#     # Load saved models
#     pattern_model = load_model('pattern_model.pkl')
#     price_model = load_model('price_prediction_model.pkl')

#     # Predict the next 5-minute closing price
#     predicted_price, next_candle_time = predict_next_candle_close(data_with_indicators, price_model)
     
#     # Call the graph_plots function
#     plot_filename = graph_plots(combined_data, price_model)
#     return render_template('index.html', 
#                            chart_url=plot_filename,  # Pass the chart URL dynamically if required
#                            next_candle_time=next_candle_time.strftime('%Y-%m-%d %H:%M:%S'),
#                            predicted_price=predicted_price,
#                            res="Prediction Successful!")  # Example status message
@app.route('/predict', methods=['GET'])
def predict():
    historical_data = load_data_from_csv(FILE_PATH)
    live_data = load_live_data('^NSEI')

    if live_data.empty:
        return render_template('index.html', res="No live data available. Market might be closed today.")

    # Rename the columns of live_data to match historical_data
    live_data.columns = ['Open', 'High', 'Low', 'Close']
    combined_data = pd.concat([historical_data, live_data], axis=0)

    # Add technical indicators
    data_with_indicators = add_technical_indicators(combined_data)

    # Prepare data for pattern recognition and price prediction
    X, y = prepare_data(data_with_indicators)
    X_price, y_price = prepare_data_for_price_prediction(data_with_indicators)  # Correct function call

    # Load saved models
    pattern_model = load_model('pattern_model.pkl')
    price_model = load_model('price_prediction_model.pkl')

    # Predict the next 5-minute closing price
    predicted_price, next_candle_time = predict_next_candle_close(data_with_indicators, price_model)
     
    # Call the graph_plots function
    plot_filename = graph_plots(combined_data, price_model)
    return render_template('index.html', 
                           chart_url=plot_filename,  # Pass the chart URL dynamically if required
                           next_candle_time=next_candle_time.strftime('%Y-%m-%d %H:%M:%S'),
                           predicted_price=predicted_price,
                           res="Prediction Successful!")  # Example status message

if __name__ == "__main__":
    app.run(debug=True)
 