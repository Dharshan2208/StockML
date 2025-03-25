import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Dropout, Input, Attention, Concatenate, LayerNormalization, MultiHeadAttention
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Force TensorFlow to use the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

print("Checking for GPU availability...")

# Function to check GPU availability with better diagnostics
def check_gpu():
    # Method 1: Check using TensorFlow
    print("Method 1: TensorFlow GPU detection...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow detected {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Memory growth enabled for {gpu.name}")

            # Use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"  Using GPU: {gpus[0].name}")

            # Set mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("  Mixed precision enabled for faster training")

            # Test GPU with a simple computation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"  GPU test computation result: {c.numpy()}")

            return gpus

        except RuntimeError as e:
            print(f"  GPU configuration error: {e}")
    else:
        print("  No GPUs detected by TensorFlow")

    # Method 2: Check if CUDA devices are available
    print("\nMethod 2: CUDA devices check...")
    try:
        device_name = tf.test.gpu_device_name()
        if device_name:
            print(f"  CUDA device found: {device_name}")
            return [device_name]
        else:
            print("  No CUDA devices found by tf.test.gpu_device_name()")
    except Exception as e:
        print(f"  CUDA detection error: {e}")

    # Method 3: Direct TensorFlow test
    print("\nMethod 3: TensorFlow device placement test...")
    try:
        if tf.test.is_built_with_cuda():
            print("  TensorFlow was built with CUDA support")
        else:
            print("  TensorFlow was NOT built with CUDA support")

        # Create a simple test tensor on GPU if possible
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0])
                print(f"  Successfully created tensor on GPU: {a.device}")
                return ['/GPU:0']
        except Exception as e:
            print(f"  Could not create tensor on GPU: {e}")
    except Exception as e:
        print(f"  TensorFlow CUDA build check error: {e}")

    print("\nNo GPU found or accessible. Using CPU instead.")
    print("If you have a GPU, try updating drivers or checking CUDA/cuDNN installation.")
    return None

# Run GPU detection
gpus = check_gpu()

# Fetch stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for the given symbol.")
    return data

# Add more features to the data
def add_features(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data.bfill(inplace=True)
    return data

# Preprocess data
def preprocess_data(data, sequence_length=60):
    data = add_features(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_10', 'EMA_10', 'Volatility', 'Momentum']])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:], scaler, data

# New function to save processed data
def save_processed_data(data, filename):
    data.to_csv(filename)
    print(f"Saved processed data to {filename}")

# Enhanced model architecture
def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GRU(256, return_sequences=True, recurrent_dropout=0.2)(inputs)
    x = Dropout(0.4)(x)
    x = GRU(128, return_sequences=True)(x)

    # Multi-head attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Concatenate()([x, attention])
    x = LayerNormalization()(x)

    x = GRU(64)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                 loss='huber',
                 metrics=['mae', 'mse'])
    return model

# Train model
def train_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Calculate appropriate batch size based on available GPU memory or default to 64
    batch_size = 128 if gpus else 64

    # Use tf.data API for more efficient data loading
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_dataset,
        epochs=150,
        validation_data=test_dataset,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )
    return model, history

# Evaluate performance
def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_test_actual = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), 4))]))[:, 0]
    y_pred_actual = scaler.inverse_transform(np.column_stack([y_pred.flatten(), np.zeros((len(y_pred), 4))]))[:, 0]

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    return y_test_actual, y_pred_actual

# New prediction function
def generate_predictions(model, data, scaler, days=30, symbol='STOCK'):
    # Convert to TensorFlow tensors for GPU acceleration
    last_sequence = tf.convert_to_tensor(data[-60:], dtype=tf.float32)
    predictions = []

    # Use GPU for batch prediction if available
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        for _ in range(days):
            # Add batch dimension
            input_seq = tf.expand_dims(last_sequence, axis=0)
            pred = model.predict(input_seq, verbose=0)
            predictions.append(pred[0,0])

            # Update sequence for next prediction
            new_seq = tf.concat([last_sequence[1:],
                                tf.reshape(tf.constant([pred[0,0], 0, 0, 0, 0], dtype=tf.float32), [1, 5])],
                                axis=0)
            last_sequence = new_seq

    # Convert predictions to actual prices
    pred_array = np.array(predictions).reshape(-1,1)
    zeros = np.zeros((len(pred_array), 4))
    full_pred = np.column_stack([pred_array, zeros])
    predictions = scaler.inverse_transform(full_pred)[:, 0]

    # Create future dates
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=days+1, freq='D')[1:]

    # Save predictions to CSV
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })
    csv_filename = f"{symbol}_future_predictions.csv"
    pred_df.to_csv(csv_filename, index=False)
    print(f"Future predictions saved to {csv_filename}")

    return future_dates, predictions

# Save model after training
def save_model(model, symbol):
    model_filename = f"{symbol}_stock_model"
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    return model_filename

# Plot results
def plot_results(y_test_actual, y_pred_actual, stock_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(y_pred_actual, label='Predicted Price')
    plt.title(f'{stock_symbol} Price Prediction (GRU-Attention Model)')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

# Time series data augmentation
def augment_data(X, y, augmentation_factor=2):
    print("Performing data augmentation...")
    X_augmented, y_augmented = [], []

    # Add original data
    X_augmented.extend(X)
    y_augmented.extend(y)

    n_samples = len(X)

    # With GPU acceleration if available
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        for i in range(augmentation_factor - 1):
            # 1. Adding jitter (random noise)
            noise_factor = 0.02
            jitter = np.random.normal(0, noise_factor, X.shape)
            X_jitter = X + jitter
            X_augmented.extend(X_jitter)
            y_augmented.extend(y)

            # 2. Time warping (scaling time steps)
            for j in range(n_samples):
                if j < n_samples - 1:  # Skip last sample to avoid index errors
                    # Randomly select a scaling factor
                    scale = np.random.uniform(0.8, 1.2)
                    X_time_warped = []

                    for k in range(X.shape[1]):
                        # Apply different scaling to each time step
                        time_steps = X[j, :, :]
                        warped = np.interp(
                            np.arange(0, len(time_steps)),
                            np.arange(0, len(time_steps)) * scale,
                            time_steps[:, 0]
                        )
                        # Reshape to match original
                        warped_reshaped = np.zeros_like(time_steps)
                        warped_reshaped[:, 0] = warped
                        # Copy other features
                        for col in range(1, time_steps.shape[1]):
                            warped_reshaped[:, col] = time_steps[:, col]

                        X_time_warped.append(warped_reshaped)

                    X_augmented.append(np.array(X_time_warped))
                    y_augmented.append(y[j])

    # Convert to numpy arrays
    return np.array(X_augmented), np.array(y_augmented)

# Main function
def main():
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ").upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # Get 3 years of data for better training

    try:
        print(f"Fetching data for {stock_symbol}...")
        data = fetch_stock_data(stock_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        # Save raw data to CSV
        raw_csv = f"{stock_symbol}_raw_data.csv"
        data.to_csv(raw_csv)
        print(f"Raw data saved to {raw_csv}")

        # Save processed data with features
        processed_data = add_features(data.copy())
        processed_csv = f"{stock_symbol}_processed_data.csv"
        processed_data.to_csv(processed_csv)
        print(f"Processed data with features saved to {processed_csv}")

        # Preprocess for model
        X_train, X_test, y_train, y_test, scaler, original_data = preprocess_data(data)

        # Apply data augmentation
        X_train_aug, y_train_aug = augment_data(X_train, y_train)
        print(f"Data augmented: Training set increased from {len(X_train)} to {len(X_train_aug)} samples")

        # Save training data samples
        train_sample = pd.DataFrame(X_train_aug[0].reshape(-1, X_train_aug.shape[2]))
        train_sample.columns = ['Close', 'SMA_10', 'EMA_10', 'Volatility', 'Momentum']
        train_sample.to_csv(f"{stock_symbol}_train_sample.csv", index=False)
        print(f"Training sample saved to {stock_symbol}_train_sample.csv")

        # Build and train model
        print("Building model with GPU acceleration if available...")
        model = build_hybrid_model((X_train_aug.shape[1], X_train_aug.shape[2]))
        model, history = train_model(model, X_train_aug, y_train_aug, X_test, y_test)

        # Save model
        model_path = save_model(model, stock_symbol)

        # Evaluate model
        y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, scaler)

        # Save test results
        test_results = pd.DataFrame({
            'Actual': y_test_actual,
            'Predicted': y_pred_actual
        })
        test_results.to_csv(f"{stock_symbol}_test_results.csv", index=False)
        print(f"Test results saved to {stock_symbol}_test_results.csv")

        # Plot results
        plot_results(y_test_actual, y_pred_actual, stock_symbol)

        # Generate future predictions
        print(f"Generating future predictions for {stock_symbol}...")
        dates, predictions = generate_predictions(model, original_data, scaler, days=30, symbol=stock_symbol)

        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions, label='Predicted Price')
        plt.title(f'{stock_symbol} Future Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{stock_symbol}_future_prediction.png")
        plt.show()

        print("\nAll data processing and prediction tasks completed successfully!")
        print(f"Files created:")
        print(f"- {raw_csv}")
        print(f"- {processed_csv}")
        print(f"- {stock_symbol}_train_sample.csv")
        print(f"- {stock_symbol}_test_results.csv")
        print(f"- {stock_symbol}_future_predictions.csv")
        print(f"- {stock_symbol}_future_prediction.png")
        print(f"- {model_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()