# Stock Price Prediction with Enhanced Deep Learning Ensemble

This project implements a sophisticated stock price prediction model using an ensemble of deep learning architectures, including LSTM, TCN (Temporal Convolutional Network), and Transformer layers with attention mechanisms. The model leverages historical stock data, technical indicators, and macroeconomic features to predict future stock prices. It is designed to handle the complexity of financial time series data and provide robust predictions.

## Features
- **Data Fetching**: Retrieves historical stock data using the `yfinance` library and macroeconomic indicators like the VIX.
- **Feature Engineering**: Incorporates technical indicators (e.g., RSI, MACD, Bollinger Bands, ATR, OBV) and lagged features.
- **Model Architecture**: Combines LSTM, TCN, and Transformer layers into an Enhanced Deep ResNet with residual connections.
- **Ensemble Learning**: Trains multiple models and aggregates predictions using performance-based weights.
- **Evaluation Metrics**: Computes RMSE, MAE, MAPE, and R² to assess model performance.
- **Visualization**: Generates plots for historical predictions and 7-day future forecasts.
- **Future Predictions**: Predicts stock prices for the next 7 days based on the latest data.

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- yfinance
- Requests (for yfinance)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/StockML.git
   cd StockML
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install torch numpy pandas matplotlib scikit-learn yfinance requests
   ```

## Usage
1. Ensure all dependencies are installed.
2. Run the script:
   ```bash
   python fucking-god.py
   ```
3. When prompted, enter a stock symbol (e.g., `AAPL` for Apple Inc,`MSFT` for Microsoft Corp).
4. The script will:
   - Fetch 5 years of historical data for the specified stock.
   - Train an ensemble of models.
   - Generate predictions for the test period and a 7-day future forecast.
   - Save results as PNG plots and a CSV file.

### Output
- **`{stock_symbol}_prediction.png`**: Plot of actual vs. predicted prices for the test period.
- **`{stock_symbol}_future_prediction.png`**: Plot of historical data and 7-day future predictions.
- **`{stock_symbol}_future_predictions.csv`**: CSV file containing future predicted prices with dates.

## Code Structure
- **Technical Indicators**: Functions to calculate ATR, RSI, OBV, MACD, and Bollinger Bands.
- **Feature Engineering**: Fetches stock and macro data, adds technical indicators and lagged features.
- **Preprocessing**: Normalizes data and creates sequences for training.
- **Model**: Defines `TCN` and `EnhancedDeepResNet` classes combining LSTM, TCN, and Transformer layers.
- **Training**: Implements ensemble training with early stopping and weighted loss.
- **Prediction**: Generates ensemble predictions and future forecasts.
- **Visualization**: Plots results and saves them to files.

## Example
```bash
Enter the stock symbol (e.g., AAPL): AAPL
Fetching data for AAPL...
Preprocessing data...
Training ensemble models...
Evaluating model...
Root Mean Squared Error: $5.23
Mean Absolute Error: $3.89
Mean Absolute Percentage Error: 2.45%
R² Score: 0.9876
Generating future predictions for AAPL...
Process completed successfully.
```

## Notes
- **Hardware**: The model benefits from a GPU (CUDA-enabled) for faster training. It falls back to CPU if no GPU is available.
- **Data**: The script fetches data from Yahoo Finance, which may occasionally fail due to rate limits or connectivity issues. Retries are implemented to handle this.
- **Customization**: Adjust hyperparameters (e.g., `sequence_length`, `num_epochs`, `ensemble_size`) in the code as needed.

## Limitations
- Predictions are based on historical patterns and may not account for sudden market events (e.g., news, earnings reports).
- Requires an internet connection to fetch data.
- Model performance depends on data quality and market conditions.

## Future Improvements
- Incorporate sentiment analysis from news or social media (e.g., X posts).
- Add more macroeconomic indicators (e.g., interest rates, GDP).
- Implement real-time data fetching and prediction updates.


## Acknowledgments
- Built with PyTorch and inspired by advancements in deep learning for time series forecasting.
- Uses `yfinance` for financial data access.

---

Feel free to modify the repository URL or any other details to suit your needs! Let me know if you'd like further adjustments.
