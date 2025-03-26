import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
from datetime import timedelta
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy
import time
from urllib3.exceptions import MaxRetryError

# --- Technical Indicator Functions ---
def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(period).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_obv(data):
    close_prices = data['Close'].to_numpy()
    volumes = data['Volume'].to_numpy()
    obv = np.zeros(len(volumes), dtype=np.float64)
    obv[0] = 0
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i-1]:
            obv[i] = obv[i-1] + volumes[i].item()
        elif close_prices[i] < close_prices[i-1]:
            obv[i] = obv[i-1] - volumes[i].item()
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd, macd.ewm(span=signal, adjust=False).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    return rolling_mean + (rolling_std * num_std), rolling_mean, rolling_mean - (rolling_std * num_std)

# --- Feature Engineering ---
def fetch_macro_data(start_date, end_date):
    try:
        vix_data = yf.download("^VIX", start=start_date, end=end_date)
        if vix_data.empty:
            raise ValueError("No VIX data fetched.")
        return pd.Series(vix_data['Close'].values.flatten(), index=vix_data.index, name='VIX')
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return pd.Series(name='VIX')

def fetch_stock_data(stock_symbol, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for {stock_symbol}")
            data = data.reset_index()
            if 'Date' not in data.columns:
                raise ValueError("No 'Date' column found in stock data after reset")
            data = data.set_index('Date', drop=True)
            print(f"Stock data index levels after reset: {data.index.nlevels}")
            return data
        except (ValueError, MaxRetryError, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Failed to fetch data for {stock_symbol} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise ValueError(f"Failed to fetch data for {stock_symbol} after {max_retries} attempts: {e}")

def add_features(data):
    data = data.copy()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['EMA_20'] = data['Close'].ewm(span=20).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['ATR'] = calculate_atr(data, 14)
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['ROC'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5) * 100
    data['RSI'] = calculate_rsi(data, 14)
    data['Volume_Change'] = data['Volume'] / data['Volume'].shift(1)
    data['OBV'] = calculate_obv(data)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)

    for lag in [1, 3, 5]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

    start_date = data.index.min().strftime('%Y-%m-%d')
    end_date = data.index.max().strftime('%Y-%m-%d')
    vix_data = fetch_macro_data(start_date, end_date).reindex(data.index, method='ffill')
    data['VIX'] = vix_data.ffill()
    return data.ffill().bfill()

def preprocess_data(data, sequence_length=60, train_size=0.8, scale=True):
    data = add_features(data)
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_10', 'EMA_20', 'Volatility', 'ATR', 'Momentum', 'ROC', 'RSI',
        'Volume_Change', 'OBV', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'VIX', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_5', 'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_5'
    ]
    data_clean = data[feature_columns].copy()
    data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in data_clean.columns:
        data_clean[col] = data_clean[col].fillna(data_clean[col].median())
    data_clean = data_clean.ffill().bfill()

    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_clean)
    else:
        scaler = None
        scaled_data = data_clean.values

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, :5])  # Predict Open, High, Low, Close, Volume

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape).astype(np.float32)
    X_train = np.concatenate([X_train, X_train_noisy])
    y_train = np.concatenate([y_train, y_train])

    return X_train, X_test, y_train, y_test, scaler, data, feature_columns

# --- Enhanced Model ---
class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=4, dropout=0.3):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim,
                            kernel_size, padding=padding, dilation=dilation)
            layers += [conv, nn.ReLU(), nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.final_conv(x)
        return x.transpose(1, 2)

class EnhancedDeepResNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 512, 1024], dropout_rate=0.3):
        super(EnhancedDeepResNet, self).__init__()
        self.input_dim = input_dim

        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims[0], num_layers=4,
                           batch_first=True, dropout=dropout_rate, bidirectional=True)

        # TCN
        self.tcn = TCN(input_dim, hidden_dims[0], num_layers=4, dropout=dropout_rate)

        # Transformer with Attention
        self.transformer_proj = nn.Linear(input_dim, hidden_dims[0])
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dims[0], nhead=8,
                                               dim_feedforward=hidden_dims[1], dropout=dropout_rate,
                                               batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=4)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dims[0], num_heads=8, dropout=dropout_rate, batch_first=True)

        # Fully connected layers with residual connections
        combined_dim = hidden_dims[0] * 2 + hidden_dims[0] * 2 + hidden_dims[0]  # LSTM (512) + TCN (256) + Transformer (256) + Attention (256) = 1280
        self.fc1 = nn.Linear(1280, hidden_dims[1])  # Corrected to match actual concatenated size
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.residual = nn.Linear(1280, hidden_dims[2])
        self.fc3 = nn.Linear(hidden_dims[2], 5)  # Predict Open, High, Low, Close, Volume

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        tcn_out = self.tcn(x)
        tcn_out = tcn_out[:, -1, :]

        x_proj = self.transformer_proj(x)
        transformer_out = self.transformer(x_proj)
        attn_out, _ = self.attention(transformer_out, transformer_out, transformer_out)
        attn_out = attn_out[:, -1, :]

        combined = torch.cat((lstm_out, tcn_out, transformer_out[:, -1, :], attn_out), dim=1)

        x = self.act(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        residual = self.residual(combined)
        x = self.act(x + residual)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# --- Training Functions ---
def weighted_mse_loss(outputs, targets):
    weights = torch.tensor([0.2, 0.2, 0.2, 0.8, 0.2], device=outputs.device)  # Higher weight for Close
    return torch.mean(weights * (outputs - targets) ** 2)

def train_model(model, X_train, y_train, X_val, y_val, device, num_epochs=300, batch_size=128, patience=50):
    criterion = weighted_mse_loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor, y_val_tensor = torch.FloatTensor(X_val).to(device), torch.FloatTensor(y_val).to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    return model, best_val_loss

def train_ensemble_model(X_train, y_train, X_val, y_val, input_dim, device, ensemble_size=15):
    models, val_losses = [], []
    for i in range(ensemble_size):
        print(f"Training model {i+1}/{ensemble_size}...")
        model = EnhancedDeepResNet(input_dim).to(device)
        model, val_loss = train_model(model, X_train, y_train, X_val, y_val, device)
        models.append(model)
        val_losses.append(val_loss)

    weights = np.exp(-np.array(val_losses))
    weights /= weights.sum()
    return models, weights

# --- Prediction Functions ---
def ensemble_predict(models, weights, X_test, device):
    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.FloatTensor(X_test).to(device)
    else:
        X_test_tensor = X_test.to(device)

    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor).cpu().numpy()
            predictions.append(pred)

    return np.average(predictions, axis=0, weights=weights)

def generate_future_predictions(models, weights, original_data, feature_columns, scaler, device, days=7):
    last_sequence = original_data[feature_columns].tail(60).copy()
    future_predictions = []
    future_data = last_sequence.copy()

    for _ in range(days):
        scaled_sequence = scaler.transform(future_data.tail(60))
        input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(device)
        pred_scaled = ensemble_predict(models, weights, input_tensor, device)[0]

        pred_array = np.zeros((1, len(feature_columns)))
        pred_array[0, :5] = pred_scaled
        pred_values = scaler.inverse_transform(pred_array)[0, :5]

        new_row = future_data.iloc[-1].copy()
        new_row['Open'] = pred_values[0]
        new_row['High'] = pred_values[1]
        new_row['Low'] = pred_values[2]
        new_row['Close'] = pred_values[3]
        new_row['Volume'] = pred_values[4]

        future_data = pd.concat([future_data, pd.DataFrame([new_row], index=[future_data.index[-1] + timedelta(days=1)])])
        future_data = add_features(future_data)

        future_predictions.append(pred_values[3])

    return np.array(future_predictions)

# --- Plotting Functions ---
def plot_results(actual, predicted, stock_symbol, metrics, dates):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Price', color='blue', linewidth=2)
    plt.plot(dates, predicted, label='Predicted Price', color='orange', linestyle='--', linewidth=2)
    plt.title(f"{stock_symbol} Stock Price Prediction\nRMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, MAPE: {metrics['MAPE']}, R²: {metrics['R2']}", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{stock_symbol}_prediction.png", dpi=300)
    plt.close()

def plot_future_predictions(dates, future_predictions, historical_data, stock_symbol):
    plt.figure(figsize=(14, 7))
    historical_dates = historical_data.index[-60:]
    historical_prices = historical_data['Close'][-60:]
    plt.plot(historical_dates, historical_prices, label='Historical Price', color='blue', linewidth=2)
    plt.plot(dates, future_predictions, label='Future Predicted Price', color='green', linestyle='--', linewidth=2)
    plt.title(f"{stock_symbol} 7-Day Future Price Prediction", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{stock_symbol}_future_prediction.png", dpi=300)
    plt.close()

# --- Main Function ---
def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")
        print(f"Fetching data for {stock_symbol}...")
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        data = fetch_stock_data(stock_symbol, start_date=start_date, end_date=end_date)

        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, scaler, original_data, feature_columns = preprocess_data(
            data, sequence_length=60, train_size=0.8, scale=True
        )

        val_size = len(X_test) // 3
        X_val, X_test = X_test[:val_size], X_test[val_size:]
        y_val, y_test = y_test[:val_size], y_test[val_size:]
        test_dates = original_data.index[-len(y_test)-val_size:-val_size]

        print("Training ensemble models...")
        models, weights = train_ensemble_model(
            X_train, y_train, X_val, y_val, input_dim=X_train.shape[2], device=device, ensemble_size=15
        )

        print("Evaluating model...")
        test_predictions = ensemble_predict(models, weights, X_test, device)
        pred_transformed = scaler.inverse_transform(np.concatenate([test_predictions, np.zeros((len(test_predictions), len(feature_columns)-5))], axis=1))[:, 3]  # Close price
        actual_transformed = scaler.inverse_transform(np.concatenate([y_test, np.zeros((len(y_test), len(feature_columns)-5))], axis=1))[:, 3]

        rmse = np.sqrt(mean_squared_error(actual_transformed, pred_transformed))
        mae = mean_absolute_error(actual_transformed, pred_transformed)
        mape = np.mean(np.abs((actual_transformed - pred_transformed) / actual_transformed)) * 100
        r2 = r2_score(actual_transformed, pred_transformed)

        metrics = {
            'RMSE': f'${rmse:.2f}', 'MAE': f'${mae:.2f}',
            'MAPE': f'{mape:.2f}%', 'R2': f'{r2:.4f}'
        }
        print(f"Root Mean Squared Error: {metrics['RMSE']}")
        print(f"Mean Absolute Error: {metrics['MAE']}")
        print(f"Mean Absolute Percentage Error: {metrics['MAPE']}")
        print(f"R² Score: {metrics['R2']}")

        plot_results(actual_transformed, pred_transformed, stock_symbol, metrics, test_dates)

        print(f"Generating future predictions for {stock_symbol}...")
        future_predictions = generate_future_predictions(models, weights, original_data, feature_columns, scaler, device, days=7)
        last_date = original_data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions})
        future_df.to_csv(f"{stock_symbol}_future_predictions.csv", index=False)

        dates = [d.date() for d in future_dates]
        plot_future_predictions(dates, future_predictions, original_data, stock_symbol)

        print("Process completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()