# model_utils.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def fetch_stock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} in range {start} to {end}")
    df = df[['Close']].reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def scale_series(series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    shaped = series.reshape(-1, 1)
    scaled = scaler.fit_transform(shaped)
    return scaled, scaler

def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(input_shape, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(max(8, units // 2)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def model_filepath_for_ticker(ticker):
    safe = ticker.upper().replace("/", "_")
    return os.path.join(MODELS_DIR, f"{safe}.h5")

def save_model_for_ticker(model, ticker):
    path = model_filepath_for_ticker(ticker)
    model.save(path)
    return path

def load_saved_model(ticker):
    path = model_filepath_for_ticker(ticker)
    if os.path.exists(path):
        # silent load
        return load_model(path)
    return None

def train_model_for_ticker(ticker, start, end, lookback=60, epochs=10, batch_size=32, future_days=0, force_retrain=False):
    """
    Train (or load) model for ticker. If a saved model exists and force_retrain=False, it will be used.
    Returns dict:
      - model
      - preds (numpy) test-set predicted prices
      - actuals (numpy)
      - dates (list of test-set dates)
      - future_preds/future_dates optional
    """
    # Fetch historical data
    df = fetch_stock(ticker, start, end)
    prices = df['Close'].values.astype('float32')

    # scale
    scaled, scaler = scale_series(prices)

    # sequences
    X, y = create_sequences(scaled, lookback=lookback)
    if len(X) < 10:
        raise ValueError("Not enough data. Increase date range or reduce lookback.")

    # train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = None
    if not force_retrain:
        model = load_saved_model(ticker)

    if model is None:
        # build & train
        model = build_lstm((X_train.shape[1], 1))
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )
        # save the trained model for future use
        try:
            save_model_for_ticker(model, ticker)
        except Exception as e:
            print("Warning: failed to save model:", e)

    # Predict on test set
    preds_scaled = model.predict(X_test).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Align test set dates
    test_dates = df['Date'].values[lookback + split_idx : lookback + split_idx + len(actuals)]
    dates = pd.to_datetime(test_dates).strftime('%Y-%m-%d').tolist()

    result = {
        'model': model,
        'preds': preds,
        'actuals': actuals,
        'dates': dates
    }

    # Recursive future forecasting if requested
    if future_days and future_days > 0:
        future_preds = []
        last_window = scaled[-lookback:].reshape(1, lookback, 1).copy()
        for _ in range(future_days):
            next_scaled = model.predict(last_window).flatten()[0]
            future_preds.append(next_scaled)
            last_window = np.append(last_window[:,1:,:], [[[next_scaled]]], axis=1)
        # inverse scale
        future_preds_unscaled = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
        last_date = df['Date'].iloc[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d').tolist()
        result['future_preds'] = future_preds_unscaled
        result['future_dates'] = future_dates

    return result
