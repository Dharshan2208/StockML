import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        Q = self.query(x)  # (batch_size, seq_len, input_dim)
        K = self.key(x)    # (batch_size, seq_len, input_dim)
        V = self.value(x)  # (batch_size, seq_len, input_dim)

        # Calculate attention scores
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)  # (batch_size, seq_len, seq_len)

        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)  # (batch_size, seq_len, seq_len)

        # Apply attention weights to values
        x = torch.matmul(attention, V)  # (batch_size, seq_len, input_dim)

        return x, attention

# Enhanced ResNet Block with additional normalization and residual connections
class EnhancedResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(EnhancedResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        # Shortcut connection with 1x1 convolution for dimensionality matching
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

        # Squeeze-and-Excitation block for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Apply SE attention
        se_weight = self.se(out)
        out = out * se_weight

        # Add shortcut connection
        out += self.shortcut(residual)
        return torch.relu(out)

# Advanced Deep ResNet Model with hybrid attention mechanisms
class DeepResNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 256, 512], num_blocks=3, dropout_rate=0.3, use_attention=True):
        super(DeepResNet, self).__init__()

        self.use_attention = use_attention
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])

        # Initial convolution
        layers = [
            nn.Conv1d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate)
        ]

        # ResNet Blocks
        in_channels = hidden_dims[0]
        for out_channels in hidden_dims[1:]:
            for _ in range(num_blocks):
                layers.append(EnhancedResNetBlock(in_channels, out_channels, dropout_rate))
                in_channels = out_channels

        # Global Average Pooling
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.features = nn.Sequential(*layers)

        # Attention layer for time series
        if use_attention:
            self.attention = SelfAttention(input_dim)
            # Transformer encoder for sequence modeling
            nhead = 4
            # Adjust input_dim to be divisible by nhead if necessary
            adjusted_dim = input_dim
            if input_dim % nhead != 0:
                adjusted_dim = (input_dim // nhead) * nhead

            encoder_layers = TransformerEncoderLayer(d_model=adjusted_dim, nhead=nhead,
                                                   dim_feedforward=adjusted_dim*4, dropout=dropout_rate)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
            # Linear layer to adjust dimensions if needed
            self.adjust_dim = nn.Identity() if input_dim == adjusted_dim else nn.Linear(input_dim, adjusted_dim)

        # Enhanced fully connected layers with skip connections
        self.fc1 = nn.Linear(hidden_dims[-1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

        # Additional skip connections
        self.skip1 = nn.Linear(hidden_dims[-1], 64)
        self.skip2 = nn.Linear(128, 32)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        if self.use_attention:
            # Apply self-attention to input
            attn_out, _ = self.attention(x)

            # Store original dimension
            original_dim = attn_out.size(-1)

            # Adjust dimensions if needed for transformer
            adjusted_dim = None
            if hasattr(self, 'adjust_dim'):
                transformer_input = self.adjust_dim(attn_out)
                adjusted_dim = transformer_input.size(-1)
                transformer_input = transformer_input.permute(1, 0, 2)
            else:
                transformer_input = attn_out.permute(1, 0, 2)

            # Apply transformer encoding
            transformer_out = self.transformer_encoder(transformer_input).permute(1, 0, 2)

            # Convert transformer output back to original dimensions if needed
            if adjusted_dim is not None and adjusted_dim != original_dim:
                # Create a projection layer on-the-fly if dimensions don't match
                projection = nn.Linear(adjusted_dim, original_dim, device=x.device)
                transformer_out = projection(transformer_out)

            # Residual connection with matching dimensions
            x = x + 0.1 * transformer_out

        # Project to initial hidden dimension
        x_proj = self.input_projection(x)
        # Reshape for 1D convolution
        x_conv = x_proj.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)

        # Extract convolutional features
        conv_features = self.features(x_conv)
        conv_features = conv_features.view(batch_size, -1)

        # Apply fully connected layers with skip connections
        x1 = self.act(self.fc1(conv_features))
        x1 = self.dropout(x1)

        skip1 = self.skip1(conv_features)

        x2 = self.act(self.fc2(x1))
        x2 = self.dropout(x2)
        x2 = x2 + skip1  # Skip connection

        skip2 = self.skip2(x1)

        x3 = self.act(self.fc3(x2))
        x3 = self.dropout(x3)
        x3 = x3 + skip2  # Skip connection

        output = self.output(x3)

        return output

# Data Preparation Functions
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for the given symbol.")
    return data

def add_features(data):
    # Basic technical indicators
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['EMA_20'] = data['Close'].ewm(span=20).mean()

    # Volatility indicators
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['ATR'] = calculate_atr(data, 14)  # Average True Range

    # Momentum indicators
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['ROC'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5) * 100  # Rate of change
    data['RSI'] = calculate_rsi(data, 14)  # Relative Strength Index

    # Volume-based indicators
    data['Volume_Change'] = data['Volume'] / data['Volume'].shift(1)
    data['OBV'] = calculate_obv(data)  # On-Balance Volume

    # MACD
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)

    # Bollinger Bands
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)

    # Fill missing values with forward fill then backward fill
    data = data.ffill()
    data = data.bfill()

    return data

# Additional Technical Indicator Functions
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()

    return macd, macd_signal

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return upper_band, rolling_mean, lower_band

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()

    return atr

def calculate_obv(data):
    """Calculate On-Balance Volume"""
    # Make sure we work with a copy to avoid modifying the original data
    close_prices = data['Close'].values
    volumes = data['Volume'].values

    # Ensure volumes is 1D
    if volumes.ndim > 1:
        volumes = volumes.flatten()

    obv = np.zeros(len(volumes))

    # First OBV value is 0
    obv[0] = 0

    # Calculate OBV values
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif close_prices[i] < close_prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]

    return pd.Series(obv, index=data.index)

# Advanced Data Augmentation Class
class TimeSeriesAugmentation:
    @staticmethod
    def jitter(x, sigma=0.03):
        """Add random noise to sequence"""
        return x + np.random.normal(0, sigma, x.shape)

    @staticmethod
    def scaling(x, sigma=0.1):
        """Apply random scaling factor"""
        factor = np.random.normal(1.0, sigma, (x.shape[0], 1, x.shape[2]))
        return x * factor

    @staticmethod
    def window_warping(x, window_ratio=0.1, scales=[0.5, 2.0]):
        """Warp a window of the time series by a random scale factor"""
        warp_scale = np.random.choice(scales)
        window_size = int(x.shape[1] * window_ratio)
        window_start = np.random.randint(0, x.shape[1] - window_size)

        x_new = x.copy()
        window = x_new[:, window_start:window_start+window_size, :]
        warped_window = np.interp(
            np.linspace(0, window.shape[1]-1, int(window.shape[1]*warp_scale)),
            np.arange(window.shape[1]),
            window.reshape(window.shape[0], window.shape[1], -1),
            axis=1
        )

        # Resize back to original window size
        warped_window = np.interp(
            np.linspace(0, warped_window.shape[1]-1, window.shape[1]),
            np.arange(warped_window.shape[1]),
            warped_window,
            axis=1
        )

        x_new[:, window_start:window_start+window_size, :] = warped_window
        return x_new

    @staticmethod
    def time_warp(x, sigma=0.2):
        """Apply random time warp"""
        from scipy.interpolate import CubicSpline

        orig_steps = np.arange(x.shape[1])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[1]//2, x.shape[2]))
        warp_steps = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            for dim in range(x.shape[2]):
                spline = CubicSpline(np.arange(x.shape[1]//2), random_warps[i, :, dim])
                warp_steps[i] = spline(orig_steps)

        ret = np.zeros_like(x)
        for i, ts in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = orig_steps * warp_steps[i]
                ret[i, :, dim] = np.interp(orig_steps, time_warp, ts[:, dim])

        return ret

    @staticmethod
    def window_slice(x, window_ratio=0.8):
        """Slice a window of the time series"""
        window_size = int(x.shape[1] * window_ratio)
        window_start = np.random.randint(0, x.shape[1] - window_size)

        return x[:, window_start:window_start+window_size, :]

# Enhanced data preprocessing with more features
def preprocess_data(data, sequence_length=60, augmentation=False):
    data = add_features(data)

    # Select columns for the model
    feature_columns = ['Close', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
                      'Volatility', 'ATR', 'Momentum', 'ROC', 'RSI',
                      'Volume_Change', 'OBV', 'MACD', 'MACD_Signal',
                      'BB_Upper', 'BB_Middle', 'BB_Lower']

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict Close price

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Apply data augmentation if enabled
    if augmentation:
        aug = TimeSeriesAugmentation()

        # Generate augmented samples
        X_jitter = aug.jitter(X_train)
        X_scaled = aug.scaling(X_train)
        X_timewarp = aug.time_warp(X_train)

        # Combine with original data
        X_train = np.vstack([X_train, X_jitter, X_scaled, X_timewarp])
        y_train = np.hstack([y_train, y_train, y_train, y_train])

        # Shuffle the augmented dataset
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

    return X_train, X_test, y_train, y_test, scaler, data, feature_columns

# Enhanced data loader with augmentation
class StockDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        return torch.FloatTensor(x_sample), torch.FloatTensor([y_sample])

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64, augment_online=False):
    # Define transforms for online augmentation
    def augment_transform(x):
        aug = TimeSeriesAugmentation()
        choice = np.random.randint(0, 4)

        if choice == 0:
            return x  # No augmentation
        elif choice == 1:
            return aug.jitter(x.reshape(1, *x.shape))[0]
        elif choice == 2:
            return aug.scaling(x.reshape(1, *x.shape))[0]
        elif choice == 3:
            return aug.time_warp(x.reshape(1, *x.shape))[0]

    # Create datasets
    transform = augment_transform if augment_online else None
    train_dataset = StockDataset(X_train, y_train, transform=transform)
    test_dataset = StockDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device, epochs=200, patience=20):
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=1.0)  # Huber loss is more robust to outliers
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler with warmup
    def warmup_cosine_schedule(step, total_steps, warmup=0.1):
        if step < warmup * total_steps:
            return float(step) / float(max(1, warmup * total_steps))
        else:
            progress = float(step - warmup * total_steps) / float(max(1, total_steps - warmup * total_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_schedule(step, epochs * len(train_loader))
    )

    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    # Training loop
    train_losses, val_losses = [], []

    # Initialize mixup
    alpha = 0.2  # Mixup interpolation coefficient

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Apply mixup augmentation
            if np.random.random() < 0.5:  # Apply mixup with 50% probability
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(X_batch.size(0)).to(device)
                mixed_X = lam * X_batch + (1 - lam) * X_batch[index]
                mixed_y = lam * y_batch + (1 - lam) * y_batch[index]

                outputs = model(mixed_X).squeeze()
                loss = criterion(outputs, mixed_y.squeeze())
            else:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch.squeeze())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                val_loss = criterion(outputs, y_batch.squeeze())
                val_epoch_loss += val_loss.item()

        # Calculate average losses
        train_loss = train_epoch_loss / len(train_loader)
        val_loss = val_epoch_loss / len(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, scaler, device, feature_columns):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()

    # Create a placeholder array for inverse transformation
    y_pred_placeholder = np.zeros((len(y_pred), len(feature_columns)))
    y_test_placeholder = np.zeros((len(y_test), len(feature_columns)))

    # Place the predictions and actual values back in their original position (first column for Close price)
    y_pred_placeholder[:, 0] = y_pred.flatten()
    y_test_placeholder[:, 0] = y_test

    # Inverse transform to get the actual prices
    y_test_actual = scaler.inverse_transform(y_test_placeholder)[:, 0]
    y_pred_actual = scaler.inverse_transform(y_pred_placeholder)[:, 0]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")

    return y_test_actual, y_pred_actual, {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def generate_future_predictions(model, original_data, scaler, device, feature_columns, days=30, symbol='STOCK'):
    model.eval()

    # Get the last sequence from the original data
    last_sequence = original_data[-60:][feature_columns].values
    last_sequence_scaled = scaler.transform(last_sequence)

    # Convert to tensor
    last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)

    # Generate predictions
    predictions = []
    prediction_dates = []
    last_date = pd.to_datetime(original_data.index[-1])

    with torch.no_grad():
        current_sequence = last_sequence_tensor

        for i in range(days):
            # Predict the next day
            pred = model(current_sequence).cpu().numpy()[0][0]
            predictions.append(pred)

            # Create the next date
            next_date = last_date + timedelta(days=i+1)
            prediction_dates.append(next_date)

            # Update the sequence
            new_row = np.zeros((1, len(feature_columns)))
            new_row[0, 0] = pred  # Set predicted Close price

            # For other features like SMA, EMA, etc. we can approximate them
            # but for simplicity we'll use the last known values
            for j in range(1, len(feature_columns)):
                new_row[0, j] = current_sequence[0, -1, j].cpu().numpy()

            # Remove oldest day and add the new prediction
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.FloatTensor(new_row).unsqueeze(1).to(device)
            ], dim=1)

    # Convert scaled predictions back to actual prices
    pred_array = np.array(predictions).reshape(-1, 1)
    placeholder = np.zeros((len(pred_array), len(feature_columns)))
    placeholder[:, 0] = pred_array.flatten()

    predictions_actual = scaler.inverse_transform(placeholder)[:, 0]

    # Create DataFrame for output
    pred_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Price': predictions_actual
    })

    # Save to CSV
    csv_filename = f"{symbol}_future_predictions.csv"
    pred_df.to_csv(csv_filename, index=False)
    print(f"Future predictions saved to {csv_filename}")

    return prediction_dates, predictions_actual

def plot_results(y_test_actual, y_pred_actual, stock_symbol, metrics=None):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Price', color='blue')
    plt.plot(y_pred_actual, label='Predicted Price', color='red')
    plt.title(f'{stock_symbol} Price Prediction (Enhanced Deep ResNet)', fontsize=16)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)

    # Add metrics to the plot if available
    if metrics:
        metrics_text = f"RMSE: ${metrics['rmse']:.2f}\nMAE: ${metrics['mae']:.2f}\nMAPE: {metrics['mape']:.2f}%\nR²: {metrics['r2']:.4f}"
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{stock_symbol}_prediction_results.png")
    plt.show()

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Stock data preparation
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ").upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # Get 5 years of data for better training

    try:
        # Fetch and preprocess data
        print(f"Fetching data for {stock_symbol}...")
        data = fetch_stock_data(stock_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        # Preprocess data with augmentation
        print("Preprocessing data and applying augmentation...")
        X_train, X_test, y_train, y_test, scaler, original_data, feature_columns = preprocess_data(
            data, sequence_length=60, augmentation=True
        )

        # Create data loaders with online augmentation
        train_loader, test_loader = create_data_loaders(
            X_train, y_train, X_test, y_test, batch_size=64, augment_online=True
        )

        # Initialize model with enhanced architecture
        model = DeepResNet(
            input_dim=X_train.shape[2],
            hidden_dims=[64, 128, 256, 512],
            num_blocks=3,
            dropout_rate=0.3,
            use_attention=True
        ).to(device)

        # Print model summary
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Train model with early stopping
        print("Training model...")
        train_losses, val_losses = train_model(
            model, train_loader, test_loader, device, epochs=200, patience=20
        )

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{stock_symbol} - Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{stock_symbol}_training_history.png")

        # Evaluate model
        print("Evaluating model...")
        y_test_actual, y_pred_actual, metrics = evaluate_model(
            model, X_test, y_test, scaler, device, feature_columns
        )

        # Plot results
        plot_results(y_test_actual, y_pred_actual, stock_symbol, metrics)

        # Generate future predictions
        print(f"Generating future predictions for {stock_symbol}...")
        dates, predictions = generate_future_predictions(
            model, original_data, scaler, device, feature_columns, days=30, symbol=stock_symbol
        )

        # Plot future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions, label='Predicted Price', color='green', marker='o')
        plt.title(f'{stock_symbol} Future Price Prediction (30 Days)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{stock_symbol}_future_prediction.png")
        plt.show()

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metrics': metrics
        }, f"{stock_symbol}_enhanced_model.pth")
        print(f"Enhanced model saved to {stock_symbol}_enhanced_model.pth")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()