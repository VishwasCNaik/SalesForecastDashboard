# ForecastingwithProphet.py

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===============================
# 1. Data Preprocessing Function
# ===============================
def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans the sales data, returns a DataFrame formatted for Prophet (columns: 'ds', 'y').

    Parameters:
    - file_path: Path to the Excel data file.

    Returns:
    - DataFrame with columns 'ds' (date) and 'y' (total sales).
    """
    df = pd.read_excel(file_path, engine="openpyxl")
    df = df.dropna(subset=['Customer ID', 'Description'])
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSales'] = df['Quantity'] * df['Price']

    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
    
    # Fill missing dates with zero sales
    daily_sales = daily_sales.set_index('ds').asfreq('D', fill_value=0).reset_index()

    return daily_sales


# ===============================
# 2. Forecasting Function
# ===============================
def forecast_with_prophet(data: pd.DataFrame, horizon: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fits a Prophet model and forecasts future sales.

    Parameters:
    - data: DataFrame with 'ds' and 'y'.
    - horizon: Number of future days to forecast.

    Returns:
    - DataFrame with forecast
    - DataFrame with actual values (last horizon days)
    - Mean Absolute Error (MAE)
    """
    train = data[:-horizon]
    test = data[-horizon:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    forecast_tail = forecast[['ds', 'yhat']].tail(horizon).set_index('ds')
    test = test.set_index('ds')

    mae = mean_absolute_error(test['y'], forecast_tail['yhat'])

    return test, forecast_tail, mae


# ===============================
# 3. Plotting Function
# ===============================
def plot_forecast(test: pd.DataFrame, forecast: pd.DataFrame, title: str = "Prophet Forecast vs Actual Sales") -> None:
    """
    Plots actual vs forecasted values using Prophet.

    Parameters:
    - test: Actual values DataFrame
    - forecast: Forecasted values DataFrame
    - title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['y'], label='Actual')
    plt.plot(forecast.index, forecast['yhat'], label='Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===============================
# 4. Main Execution
# ===============================
if __name__ == "__main__":
    FILE_PATH = r"V:\GitHub\SalesForecastDashboard\online_retail_II.xlsx"

    logging.info("Preprocessing data...")
    prophet_df = preprocess_data(FILE_PATH)

    logging.info("Training Prophet model and forecasting...")
    test, forecast, mae = forecast_with_prophet(prophet_df, horizon=30)

    logging.info(f"Mean Absolute Error: {mae:.2f}")

    plot_forecast(test, forecast)
