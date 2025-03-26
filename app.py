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
import datetime
from datetime import timedelta
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy
import os
import json
import joblib

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

class EnhancedDeepResNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 256, 512], num_blocks=3, dropout_rate=0.3):
        super(EnhancedDeepResNet, self).__init__()

        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])

        # Multi-head self-attention with improved architecture
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4 if input_dim % 4 == 0 else 1,
            dropout=dropout_rate,
            batch_first=True
        )

        # Convolutional feature extraction branch
        conv_layers = [
            nn.Conv1d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate)
        ]

        # Enhanced ResNet blocks with increased depth
        in_channels = hidden_dims[0]
        for out_channels in hidden_dims[1:]:
            for _ in range(num_blocks):
                conv_layers.append(EnhancedResNetBlock(in_channels, out_channels, dropout_rate))
                in_channels = out_channels

        # Global pooling
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv_features = nn.Sequential(*conv_layers)

        # Transformer branch for sequential modeling
        transformer_dim = input_dim - (input_dim % 8) if input_dim % 8 != 0 else input_dim
        self.transformer_projection = nn.Linear(input_dim, transformer_dim) if input_dim != transformer_dim else nn.Identity()

        encoder_layers = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=8,
            dim_feedforward=transformer_dim*4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=3)
        self.transformer_pooling = nn.AdaptiveAvgPool1d(1)

        # Bi-directional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )

        # Fully connected layers with enhanced skip connections
        self.fc1 = nn.Linear(hidden_dims[-1] + hidden_dims[0]*2 + transformer_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # Skip connections for better gradient flow
        self.skip1 = nn.Linear(hidden_dims[-1] + hidden_dims[0]*2 + transformer_dim, 128)
        self.skip2 = nn.Linear(256, 64)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()  # Swish activation function for better gradient flow

        # Layer normalization for better training stability
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(64)

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        # 1. Self-attention branch
        attn_output, _ = self.self_attention(x, x, x)

        # 2. Convolutional branch
        x_proj = self.input_projection(x)
        x_conv = x_proj.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        conv_output = self.conv_features(x_conv)
        conv_output = conv_output.view(batch_size, -1)

        # 3. Transformer branch
        transformer_input = self.transformer_projection(x)
        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.permute(0, 2, 1)  # (batch, dim, seq_len)
        transformer_output = self.transformer_pooling(transformer_output).view(batch_size, -1)

        # 4. LSTM branch
        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]  # Take the last time step output

        # Concatenate all features from different branches
        combined_features = torch.cat([conv_output, lstm_output, transformer_output], dim=1)

        # Apply fully connected layers with skip connections and normalization
        x1 = self.act(self.norm1(self.fc1(combined_features)))
        x1 = self.dropout(x1)

        skip1 = self.skip1(combined_features)

        x2 = self.act(self.norm2(self.fc2(x1)))
        x2 = self.dropout(x2)
        x2 = x2 + skip1  # Skip connection

        skip2 = self.skip2(x1)

        x3 = self.act(self.norm3(self.fc3(x2)))
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

def time_warp(data, sigma=0.2, knot=4):
    """Apply time warping augmentation to time series data"""
    orig_steps = np.arange(data.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], knot+2))
    warp_steps = np.zeros((data.shape[0], data.shape[1]))

    for i in range(data.shape[0]):
        knot_loc = np.random.randint(low=1, high=data.shape[1]-1, size=knot)
        knot_loc = np.sort(np.concatenate([np.array([0]), knot_loc, np.array([data.shape[1]-1])]))
        warper = np.interp(orig_steps, knot_loc, random_warps[i])
        warp_steps[i] = np.cumsum(warper)

    warped = np.zeros_like(data)
    for i in range(data.shape[0]):
        for dim in range(data.shape[2]):
            time_series = data[i, :, dim]
            warped[i, :, dim] = np.interp(warp_steps[i], orig_steps, time_series)

    return warped

def advanced_mixup(x, y, alpha=0.2):
    """
    Enhanced Mixup augmentation that adapts alpha based on similarity of samples

    Args:
        x: Input data tensor or numpy array
        y: Target tensor or numpy array
        alpha: Mixup alpha parameter controlling interpolation strength

    Returns:
        Mixed input, Mixed target
    """
    # Determine if inputs are PyTorch tensors or NumPy arrays
    is_tensor = torch.is_tensor(x)

    if is_tensor:
        # Convert to numpy for processing
        device = x.device
        dtype = x.dtype  # Store original dtype
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
    else:
        x_np = x
        y_np = y
        dtype = torch.float32  # Default dtype

    batch_size = x_np.shape[0]

    # Calculate sample similarities in the batch
    similarities = np.zeros(batch_size)
    for i in range(batch_size):
        sample_distances = np.array([np.linalg.norm(x_np[i] - x_np[j]) for j in range(batch_size)])
        similarities[i] = 1.0 / (np.mean(sample_distances) + 1e-10)  # Add epsilon to avoid division by zero

    # Normalize similarities to [0, 1]
    if np.max(similarities) > np.min(similarities):
        similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))
    else:
        similarities = np.ones_like(similarities) * 0.5

    # Adjust alpha based on similarity (more similar = stronger mixing)
    # Ensure alpha is always > 0 with a minimum value
    adaptive_alpha = alpha + similarities * alpha
    adaptive_alpha = np.maximum(adaptive_alpha, 0.01)  # Ensure minimum value

    # Generate mixup weights from beta distribution
    lam = np.random.beta(adaptive_alpha, adaptive_alpha, size=batch_size)
    lam = np.expand_dims(lam, axis=(1, 2))

    # Generate random indices for mixing
    indices = np.random.permutation(batch_size)

    # Mix inputs and targets in numpy
    mixed_x_np = lam * x_np + (1 - lam) * x_np[indices]
    mixed_y_np = lam.reshape(-1, 1) * y_np + (1 - lam.reshape(-1, 1)) * y_np[indices]

    # Convert back to tensor if the input was a tensor
    if is_tensor:
        # Ensure the data type is consistent with the input
        mixed_x = torch.tensor(mixed_x_np, dtype=dtype, device=device)
        mixed_y = torch.tensor(mixed_y_np, dtype=dtype, device=device)
    else:
        mixed_x = mixed_x_np
        mixed_y = mixed_y_np

    return mixed_x, mixed_y

def freq_mask(data, num_masks=1, max_width=0.1):
    """Apply frequency domain masking for better generalization"""
    # Convert to frequency domain
    fft_data = np.fft.rfft(data, axis=1)

    for i in range(data.shape[0]):
        for _ in range(num_masks):
            mask_width = int(max_width * fft_data.shape[1])
            if mask_width < 1:
                mask_width = 1

            # Random starting point for the mask
            mask_start = np.random.randint(0, fft_data.shape[1] - mask_width + 1)
            fft_data[i, mask_start:mask_start+mask_width, :] = 0

    # Convert back to time domain
    augmented_data = np.fft.irfft(fft_data, n=data.shape[1], axis=1)

    return augmented_data

def cutout(data, num_cutouts=2, cutout_size=0.1):
    """Apply cutout augmentation to the time series data"""
    augmented_data = data.copy()
    time_length = data.shape[1]
    cutout_length = int(time_length * cutout_size)

    if cutout_length < 1:
        cutout_length = 1

    for i in range(data.shape[0]):
        for _ in range(num_cutouts):
            start_idx = np.random.randint(0, time_length - cutout_length + 1)
            # Use the average value or neighboring points for more natural filling
            if start_idx > 0 and start_idx + cutout_length < time_length:
                fill_value = (data[i, start_idx-1, :] + data[i, start_idx+cutout_length, :]) / 2
            else:
                fill_value = np.mean(data[i], axis=0)

            augmented_data[i, start_idx:start_idx+cutout_length, :] = fill_value

    return augmented_data

def window_slice(data, reduce_ratio=0.7):
    """Extract a random continuous window from the time series"""
    window_size = int(data.shape[1] * reduce_ratio)
    if window_size < 2:
        return data

    window_starts = np.random.randint(0, data.shape[1] - window_size + 1, size=data.shape[0])

    sliced_data = np.zeros((data.shape[0], window_size, data.shape[2]))
    for i, window_start in enumerate(window_starts):
        sliced_data[i] = data[i, window_start:window_start+window_size, :]

    # Resize back to original length
    augmented_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for dim in range(data.shape[2]):
            augmented_data[i, :, dim] = np.interp(
                np.linspace(0, window_size-1, data.shape[1]),
                np.arange(window_size),
                sliced_data[i, :, dim]
            )

    return augmented_data

def apply_augmentations(X, y=None, strong_augment=False):
    """Apply multiple augmentation techniques to batch data"""
    batch_X = X.copy()
    batch_y = y.copy() if y is not None else None

    # Basic augmentations with lower probability
    if np.random.rand() < 0.5:
        batch_X = TimeSeriesAugmentation.jitter(batch_X, sigma=0.03)

    if np.random.rand() < 0.5:
        batch_X = TimeSeriesAugmentation.scaling(batch_X, sigma=0.1)

    # More diverse augmentations with higher intensity for strong augmentation
    if strong_augment:
        # Apply advanced augmentations
        if np.random.rand() < 0.3:
            batch_X = time_warp(batch_X, sigma=0.3, knot=4)

        if np.random.rand() < 0.3:
            batch_X = freq_mask(batch_X, num_masks=2, max_width=0.15)

        if np.random.rand() < 0.3:
            batch_X = cutout(batch_X, num_cutouts=3, cutout_size=0.15)

        if np.random.rand() < 0.3:
            batch_X = window_slice(batch_X, reduce_ratio=0.6)

    return batch_X, batch_y

def train_ensemble_model(X_train, y_train, X_val, y_val, input_dim, device, ensemble_size=3):
    """Train an ensemble of models with different initializations and architectures"""
    models = []
    val_losses = []

    # Architectures to use for ensemble diversity
    architectures = [
        {'hidden_dims': [64, 128, 256, 512], 'dropout_rate': 0.3},
        {'hidden_dims': [32, 64, 128, 256, 512], 'dropout_rate': 0.4},
        {'hidden_dims': [128, 256, 384, 512], 'dropout_rate': 0.25}
    ]

    # Train models with different configurations
    for i in range(ensemble_size):
        print(f"Training ensemble model {i+1}/{ensemble_size}")

        # Use different architectures for different ensemble members
        config = architectures[i % len(architectures)]

        # Initialize model with the specific configuration
        model = EnhancedDeepResNet(
            input_dim=input_dim,
            hidden_dims=config['hidden_dims'],
            num_blocks=3,
            dropout_rate=config['dropout_rate']
        ).to(device)

        # Vary optimization parameters for diversity
        lr = 5e-4 * (1.0 + 0.1 * (i % 3))
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Use different loss functions for diversity
        if i % 3 == 0:
            criterion = nn.HuberLoss(delta=1.0)
        elif i % 3 == 1:
            criterion = nn.MSELoss()
        else:
            criterion = nn.SmoothL1Loss()

        # Train the model
        val_loss = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            optimizer,
            criterion,
            device,
            num_epochs=40,
            batch_size=32,
            patience=10,
            apply_augmentation=True,
            strong_augment=(i % 2 == 0)  # Alternate between strong and regular augmentation
        )

        models.append(model)
        val_losses.append(val_loss)

    # Create weights for ensemble averaging based on validation performance
    inverse_losses = [1.0/loss for loss in val_losses]
    total = sum(inverse_losses)
    weights = [loss/total for loss in inverse_losses]

    return models, weights

def ensemble_predict(models, weights, X, device):
    """
    Generate predictions using weighted ensemble averaging

    Args:
        models: List of trained models
        weights: List of model weights
        X: Input data
        device: PyTorch device

    Returns:
        Weighted ensemble predictions
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            prediction = model(X_tensor).cpu().numpy()
            predictions.append(prediction)

    # Apply weights to each model's predictions
    ensemble_pred = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        ensemble_pred += weights[i] * pred

    return ensemble_pred

def train_model(model, X_train, y_train, X_val, y_val, optimizer, criterion, device, num_epochs=50, batch_size=32, patience=10, apply_augmentation=True, strong_augment=False):
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        steps_per_epoch=int(np.ceil(len(X_train) / batch_size)),
        epochs=num_epochs,
        anneal_strategy='cos'
    )

    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation data
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            # Apply data augmentation
            if apply_augmentation:
                X_batch_np = X_batch.numpy()
                y_batch_np = y_batch.numpy()

                X_batch_aug, y_batch_aug = apply_augmentations(X_batch_np, y_batch_np, strong_augment)
                X_batch = torch.FloatTensor(X_batch_aug)
                y_batch = torch.FloatTensor(y_batch_aug)

                # Apply mixup augmentation (on the PyTorch tensors)
                X_batch, y_batch = advanced_mixup(X_batch, y_batch)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print training progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_val_loss

def preprocess_data(data, sequence_length=60, train_size=0.8, scale=True):
    """
    Preprocess data with enhanced feature engineering

    Args:
        data: DataFrame with OHLCV data
        sequence_length: Length of input sequences
        train_size: Proportion of data to use for training
        scale: Whether to scale the data

    Returns:
        X_train, X_test, y_train, y_test, scaler, original_data, feature_columns
    """
    data = add_features(data)

    # Select columns for the model - expanded set of features
    feature_columns = [
        'Close', 'Volume', 'Open', 'High', 'Low',  # Basic OHLCV
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100',  # Multiple SMAs
        'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',  # Multiple EMAs
        'Volatility', 'ATR', 'Momentum', 'ROC', 'RSI',  # Technical indicators
        'Volume_Change', 'OBV', 'MACD', 'MACD_Signal',  # Volume and momentum
        'BB_Upper', 'BB_Middle', 'BB_Lower'  # Bollinger Bands
    ]

    # Ensure all features exist in the data
    available_features = [col for col in feature_columns if col in data.columns]

    # Clean data: replace inf, -inf with NaN, then fill NaN with appropriate values
    data_clean = data[available_features].copy()

    # Replace inf/-inf with NaN
    data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    # For each column, fill NaN with the column median
    for col in data_clean.columns:
        col_median = data_clean[col].median()
        # Fix pandas warning by using a different approach
        data_clean.loc[:, col] = data_clean[col].fillna(col_median)

    # Additional check for any remaining NaN values
    if data_clean.isna().any().any():
        # Forward fill then backward fill any remaining NaNs
        data_clean = data_clean.ffill().bfill()

    # Normalize data
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_clean)
    else:
        scaler = None
        scaled_data = data_clean.values

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict Close price (first column)

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    # Split into train and test sets
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler, data, available_features

def generate_future_predictions(models, weights, data, feature_columns, scaler, device, days=30):
    """
    Generate future price predictions using an ensemble of models

    Args:
        models: List of trained models in the ensemble
        weights: List of weights for each model
        data: Original DataFrame with all data
        feature_columns: List of feature column names
        scaler: Fitted scaler for inverse transformation
        device: PyTorch device
        days: Number of days to predict into the future

    Returns:
        Array of predicted prices
    """
    # Get the last sequence from the data
    sequence_length = 60  # Same as used in training

    # Extract the relevant features and scale
    last_sequence = data[feature_columns].iloc[-sequence_length:].values
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)

    # Initialize predictions array
    predictions = []

    # Create a copy of the last sequence that we'll update with each prediction
    current_sequence = last_sequence_scaled.copy()

    # Generate predictions for the specified number of days
    for i in range(days):
        # Convert current sequence to tensor and get ensemble prediction
        X_pred = np.array([current_sequence])
        pred = ensemble_predict(models, weights, X_pred, device)

        # Save the prediction
        predictions.append(pred[0, 0])

        # Update the sequence: remove the oldest day, add the newest prediction
        # For simplicity, we'll only update the Close price and keep other features constant
        new_point = current_sequence[-1].copy()
        new_point[0] = pred[0, 0]  # Update Close price

        # Shift the sequence and add the new point
        current_sequence = np.vstack([current_sequence[1:], new_point])

    # Inverse transform predictions to get actual prices
    inv_predictions = []
    for pred in predictions:
        # Create a full feature vector with the prediction in the Close position
        pred_vector = np.zeros((1, len(feature_columns)))
        pred_vector[0, 0] = pred  # Close price is the first feature

        # Inverse transform and extract the Close price
        inv_pred = scaler.inverse_transform(pred_vector)[0, 0]
        inv_predictions.append(inv_pred)

    return np.array(inv_predictions)

def plot_results(actual, predicted, stock_symbol, metrics):
    """
    Create visualizations of actual vs predicted stock prices

    Args:
        actual: Array of actual prices
        predicted: Array of predicted prices
        stock_symbol: Stock ticker symbol
        metrics: Dictionary of performance metrics
    """
    plt.figure(figsize=(14, 7))

    # Plot actual vs predicted
    plt.plot(actual, label='Actual Price', color='blue', alpha=0.7, linewidth=2)
    plt.plot(predicted, label='Predicted Price', color='red', alpha=0.7, linewidth=2)

    # Calculate and plot error bands (prediction intervals)
    mae_value = float(metrics['MAE'].replace('$', ''))
    error_band = mae_value * 1.96  # 95% confidence interval approximation

    plt.fill_between(
        range(len(predicted)),
        predicted - error_band,
        predicted + error_band,
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )

    # Add markers at turning points (local extrema)
    def find_turning_points(prices, window=5):
        turning_points = []
        for i in range(window, len(prices) - window):
            if (prices[i] > max(prices[i-window:i]) and prices[i] > max(prices[i+1:i+window+1])):
                turning_points.append((i, prices[i], 'peak'))  # Peak
            if (prices[i] < min(prices[i-window:i]) and prices[i] < min(prices[i+1:i+window+1])):
                turning_points.append((i, prices[i], 'valley'))  # Valley
        return turning_points

    # Mark turning points on actual data
    actual_turning_points = find_turning_points(actual)
    for i, price, point_type in actual_turning_points:
        marker = '^' if point_type == 'peak' else 'v'
        color = 'green' if point_type == 'peak' else 'purple'
        plt.scatter(i, price, color=color, marker=marker, s=80, alpha=0.7,
                   edgecolors='black', linewidths=1)

    # Add linear regression trendline
    x = np.arange(len(actual))
    z = np.polyfit(x, actual, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "g--", alpha=0.7, linewidth=1.5, label='Trend')

    # Enhance the appearance
    plt.title(f'{stock_symbol} Price Prediction (Enhanced Ensemble Model)', fontsize=16, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add metrics to the plot
    metrics_text = (
        f"RMSE: {metrics['RMSE']}\n"
        f"MAE: {metrics['MAE']}\n"
        f"MAPE: {metrics['MAPE']}\n"
        f"R²: {metrics['R2']}"
    )

    plt.figtext(0.02, 0.02, metrics_text, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    # Save the visualization
    plt.savefig(f"{stock_symbol}_prediction_results.png", dpi=300, bbox_inches='tight')
    print(f"Results visualization saved to {stock_symbol}_prediction_results.png")

def plot_future_predictions(dates, predictions, stock_symbol):
    """
    Create visualizations of future price predictions

    Args:
        dates: List of future dates
        predictions: Array of predicted prices
        stock_symbol: Stock ticker symbol
    """
    plt.figure(figsize=(14, 7))

    # Plot predictions with confidence interval
    std_dev = predictions.std() * 1.5  # Estimated standard deviation for confidence interval

    plt.plot(dates, predictions, label='Predicted Price', color='green',
             marker='o', markersize=6, linewidth=2)

    plt.fill_between(
        dates,
        predictions - std_dev,
        predictions + std_dev,
        color='green',
        alpha=0.2,
        label='Confidence Interval'
    )

    # Find trend direction
    if len(predictions) > 1:
        trend = (predictions[-1] - predictions[0]) / predictions[0] * 100
        trend_direction = "Upward" if trend > 0 else "Downward"
        trend_text = f"{trend_direction} Trend: {abs(trend):.2f}%"

        # Add annotation for trend
        plt.annotate(
            trend_text,
            xy=(dates[len(dates)//2], predictions.max()),
            xytext=(dates[len(dates)//2], predictions.max() * 1.1),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

    # Enhance the appearance
    plt.title(f'{stock_symbol} Future Price Prediction (30 Days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)

    # Add key price levels
    min_price = predictions.min()
    max_price = predictions.max()

    plt.axhline(y=min_price, color='r', linestyle=':', alpha=0.7,
               label=f'Support: ${min_price:.2f}')
    plt.axhline(y=max_price, color='b', linestyle=':', alpha=0.7,
               label=f'Resistance: ${max_price:.2f}')

    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    # Save the visualization
    plt.savefig(f"{stock_symbol}_future_prediction.png", dpi=300, bbox_inches='tight')
    print(f"Future prediction visualization saved to {stock_symbol}_future_prediction.png")

def save_ensemble_model(models, weights, scaler, feature_columns, metrics, stock_symbol):
    """
    Save the ensemble model and related data for later use

    Args:
        models: List of trained models
        weights: List of model weights
        scaler: The fitted scaler
        feature_columns: List of feature column names
        metrics: Performance metrics dictionary
        stock_symbol: Stock ticker symbol
    """
    # Create a directory for the model if it doesn't exist
    model_dir = f"{stock_symbol}_ensemble_model"
    os.makedirs(model_dir, exist_ok=True)

    # Save each model in the ensemble
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"{model_dir}/model_{i}.pth")

    # Save weights, scaler, feature names and other metadata
    metadata = {
        'weights': weights,
        'feature_columns': feature_columns,
        'metrics': metrics,
        'model_architecture': {
            'type': 'EnhancedDeepResNet',
            'models_count': len(models)
        },
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save scaler separately using joblib (better for sklearn objects)
    joblib.dump(scaler, f"{model_dir}/scaler.joblib")

    # Save metadata as JSON
    with open(f"{model_dir}/metadata.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metadata = metadata.copy()
        serializable_metadata['weights'] = [float(w) for w in metadata['weights']]
        json.dump(serializable_metadata, f, indent=4)

    # Create a README file with model information
    readme_content = f"""# {stock_symbol} Ensemble Stock Prediction Model

## Model Information
- Stock Symbol: {stock_symbol}
- Created: {metadata['timestamp']}
- Architecture: Enhanced Deep Learning Ensemble
- Number of Models: {len(models)}

## Performance Metrics
- RMSE: {metrics['RMSE']}
- MAE: {metrics['MAE']}
- MAPE: {metrics['MAPE']}
- R² Score: {metrics['R2']}

## Files
- `model_*.pth`: Individual model weights
- `scaler.joblib`: Feature scaler
- `metadata.json`: Model configuration and metadata

## Features Used
{', '.join(feature_columns)}

## Usage
Load this model using the load_ensemble_model function.
"""

    with open(f"{model_dir}/README.md", 'w') as f:
        f.write(readme_content)

    print(f"Enhanced ensemble model saved to {model_dir}/")
    return model_dir

def load_ensemble_model(model_dir, device):
    """
    Load a saved ensemble model

    Args:
        model_dir: Directory containing the saved model files
        device: PyTorch device to load the model to

    Returns:
        models, weights, scaler, feature_columns
    """
    # Load metadata
    with open(f"{model_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)

    weights = metadata['weights']
    feature_columns = metadata['feature_columns']
    models_count = metadata['model_architecture']['models_count']

    # Load scaler
    scaler = joblib.load(f"{model_dir}/scaler.joblib")

    # Load models
    models = []
    for i in range(models_count):
        # Create model with appropriate architecture
        model = EnhancedDeepResNet(
            input_dim=len(feature_columns),
            hidden_dims=[64, 128, 256, 512],
            num_blocks=3,
            dropout_rate=0.3
        ).to(device)

        # Load state dict
        model.load_state_dict(torch.load(f"{model_dir}/model_{i}.pth"))
        model.eval()  # Set to evaluation mode
        models.append(model)

    print(f"Loaded ensemble model from {model_dir}/")
    return models, weights, scaler, feature_columns

def main():
    try:
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Get user input for stock symbol
        stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")

        # Fetch data
        print(f"Fetching data for {stock_symbol}...")
        data = fetch_stock_data(stock_symbol, start_date=None, end_date=None)

        # Preprocess data
        print("Preprocessing data and applying augmentation...")
        X_train, X_test, y_train, y_test, scaler, original_data, feature_columns = preprocess_data(
            data, sequence_length=60, train_size=0.8, scale=True
        )

        # Train ensemble of models
        print("Training ensemble models...")
        models, weights = train_ensemble_model(
            X_train, y_train,
            X_test[:len(X_test)//3], y_test[:len(y_test)//3],  # Use first 1/3 of test data as validation
            input_dim=X_train.shape[2],
            device=device,
            ensemble_size=3
        )

        # Evaluate model performance
        print("Evaluating model...")
        test_predictions = ensemble_predict(models, weights, X_test, device)

        # Reshape test predictions
        test_predictions = test_predictions.reshape(-1)
        y_test = y_test.reshape(-1)

        # Inverse transform predictions and actual values
        features_array = np.zeros((len(test_predictions), len(feature_columns)))
        features_array[:, 0] = test_predictions  # Assuming the first column is the target
        pred_transformed = scaler.inverse_transform(features_array)[:,0]

        features_array[:, 0] = y_test
        actual_transformed = scaler.inverse_transform(features_array)[:,0]

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(actual_transformed, pred_transformed))
        mae = mean_absolute_error(actual_transformed, pred_transformed)
        mape = np.mean(np.abs((actual_transformed - pred_transformed) / actual_transformed)) * 100
        r2 = r2_score(actual_transformed, pred_transformed)

        metrics = {
            'RMSE': f'${rmse:.2f}',
            'MAE': f'${mae:.2f}',
            'MAPE': f'{mape:.2f}%',
            'R2': f'{r2:.4f}'
        }

        print(f"Root Mean Squared Error: {metrics['RMSE']}")
        print(f"Mean Absolute Error: {metrics['MAE']}")
        print(f"Mean Absolute Percentage Error: {metrics['MAPE']}")
        print(f"R² Score: {metrics['R2']}")

        # Plot test results
        plot_results(actual_transformed, pred_transformed, stock_symbol, metrics)

        # Generate future predictions
        print(f"Generating future predictions for {stock_symbol}...")
        future_predictions = generate_future_predictions(models, weights, original_data, feature_columns, scaler, device, days=30)

        # Create dates for future predictions
        last_date = original_data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]

        # Save future predictions to CSV
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions
        })
        future_df.to_csv(f"{stock_symbol}_future_predictions.csv", index=False)

        # Plot future predictions
        dates = [d.date() for d in future_dates]
        plot_future_predictions(dates, future_predictions, stock_symbol)

        # Save the enhanced ensemble model
        save_ensemble_model(models, weights, scaler, feature_columns, metrics, stock_symbol)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()