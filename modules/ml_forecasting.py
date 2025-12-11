"""
ML Forecasting Module

Price prediction using statistical and ML-based approaches:
- Simple moving average forecasting
- ARIMA-style predictions
- Neural network forecasting (when TensorFlow available)
- Confidence intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class ForecastResult:
    """Result of price forecasting."""
    predictions: np.ndarray
    dates: pd.DatetimeIndex
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence: float
    method: str


def forecast_naive(
    data: pd.DataFrame,
    horizon: int = 5
) -> ForecastResult:
    """
    Simple naive forecast (last value carried forward).
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
    
    Returns:
        ForecastResult
    """
    last_price = data['Close'].iloc[-1]
    predictions = np.full(horizon, last_price)
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
    
    # Confidence based on recent volatility
    returns = data['Close'].pct_change().dropna()
    std = returns.std()
    
    lower = predictions * (1 - 2 * std * np.sqrt(np.arange(1, horizon + 1)))
    upper = predictions * (1 + 2 * std * np.sqrt(np.arange(1, horizon + 1)))
    
    return ForecastResult(
        predictions=predictions,
        dates=future_dates,
        lower_bound=lower,
        upper_bound=upper,
        confidence=0.4,
        method='Naive'
    )


def forecast_moving_average(
    data: pd.DataFrame,
    horizon: int = 5,
    window: int = 20
) -> ForecastResult:
    """
    Moving average based forecast with trend extrapolation.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
        window: Moving average window
    
    Returns:
        ForecastResult
    """
    close = data['Close']
    ma = close.rolling(window).mean()
    
    # Calculate trend
    recent_ma = ma.tail(10).values
    if len(recent_ma) >= 2:
        trend = (recent_ma[-1] - recent_ma[0]) / len(recent_ma)
    else:
        trend = 0
    
    last_ma = ma.iloc[-1]
    predictions = np.array([last_ma + trend * (i + 1) for i in range(horizon)])
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
    
    # Confidence intervals
    returns = close.pct_change().dropna()
    std = returns.std()
    
    lower = predictions * (1 - 1.96 * std * np.sqrt(np.arange(1, horizon + 1)))
    upper = predictions * (1 + 1.96 * std * np.sqrt(np.arange(1, horizon + 1)))
    
    return ForecastResult(
        predictions=predictions,
        dates=future_dates,
        lower_bound=lower,
        upper_bound=upper,
        confidence=0.5,
        method='Moving Average'
    )


def forecast_exponential_smoothing(
    data: pd.DataFrame,
    horizon: int = 5,
    alpha: float = 0.3,
    beta: float = 0.1
) -> ForecastResult:
    """
    Double exponential smoothing (Holt's method) forecast.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
        alpha: Level smoothing factor
        beta: Trend smoothing factor
    
    Returns:
        ForecastResult
    """
    close = data['Close'].values
    n = len(close)
    
    # Initialize
    level = close[0]
    trend = close[1] - close[0] if n > 1 else 0
    
    # Apply double exponential smoothing
    for i in range(1, n):
        prev_level = level
        level = alpha * close[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    
    # Forecast
    predictions = np.array([level + trend * (i + 1) for i in range(horizon)])
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
    
    # Confidence intervals
    returns = data['Close'].pct_change().dropna()
    std = returns.std()
    
    lower = predictions * (1 - 1.96 * std * np.sqrt(np.arange(1, horizon + 1)))
    upper = predictions * (1 + 1.96 * std * np.sqrt(np.arange(1, horizon + 1)))
    
    return ForecastResult(
        predictions=predictions,
        dates=future_dates,
        lower_bound=lower,
        upper_bound=upper,
        confidence=0.55,
        method='Exponential Smoothing'
    )


def forecast_linear_regression(
    data: pd.DataFrame,
    horizon: int = 5,
    lookback: int = 30
) -> ForecastResult:
    """
    Linear regression based forecast.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
        lookback: Number of periods for regression
    
    Returns:
        ForecastResult
    """
    close = data['Close'].tail(lookback).values
    x = np.arange(len(close))
    
    # Fit linear regression
    coeffs = np.polyfit(x, close, 1)
    slope, intercept = coeffs
    
    # Forecast
    future_x = np.arange(len(close), len(close) + horizon)
    predictions = slope * future_x + intercept
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
    
    # Calculate residual standard error
    fitted = slope * x + intercept
    residuals = close - fitted
    std_err = np.std(residuals)
    
    lower = predictions - 1.96 * std_err * np.sqrt(1 + ((future_x - x.mean())**2).sum() / ((x - x.mean())**2).sum())
    upper = predictions + 1.96 * std_err * np.sqrt(1 + ((future_x - x.mean())**2).sum() / ((x - x.mean())**2).sum())
    
    return ForecastResult(
        predictions=predictions,
        dates=future_dates,
        lower_bound=lower,
        upper_bound=upper,
        confidence=0.45,
        method='Linear Regression'
    )


def forecast_ensemble(
    data: pd.DataFrame,
    horizon: int = 5
) -> ForecastResult:
    """
    Ensemble forecast combining multiple methods.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
    
    Returns:
        ForecastResult with averaged predictions
    """
    # Get forecasts from multiple methods
    forecasts = [
        forecast_naive(data, horizon),
        forecast_moving_average(data, horizon),
        forecast_exponential_smoothing(data, horizon),
        forecast_linear_regression(data, horizon)
    ]
    
    # Weight by confidence
    total_confidence = sum(f.confidence for f in forecasts)
    weights = [f.confidence / total_confidence for f in forecasts]
    
    # Weighted average of predictions
    predictions = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)
    
    for f, w in zip(forecasts, weights):
        predictions += w * f.predictions
        lower += w * f.lower_bound
        upper += w * f.upper_bound
    
    future_dates = forecasts[0].dates
    
    return ForecastResult(
        predictions=predictions,
        dates=future_dates,
        lower_bound=lower,
        upper_bound=upper,
        confidence=0.6,  # Ensemble typically more reliable
        method='Ensemble'
    )


def try_lstm_forecast(
    data: pd.DataFrame,
    horizon: int = 5,
    sequence_length: int = 60
) -> Optional[ForecastResult]:
    """
    Attempt LSTM forecast if TensorFlow is available.
    Falls back to None if not available.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
        sequence_length: LSTM sequence length
    
    Returns:
        ForecastResult or None
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        
        close = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)
        
        # Prepare sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i, 0])
            y.append(scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # Forecast
        last_sequence = scaled[-sequence_length:].reshape((1, sequence_length, 1))
        predictions_scaled = []
        
        for _ in range(horizon):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            predictions_scaled.append(pred)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred
        
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
        
        # Confidence intervals
        returns = data['Close'].pct_change().dropna()
        std = returns.std()
        
        lower = predictions * (1 - 1.96 * std * np.sqrt(np.arange(1, horizon + 1)))
        upper = predictions * (1 + 1.96 * std * np.sqrt(np.arange(1, horizon + 1)))
        
        return ForecastResult(
            predictions=predictions,
            dates=future_dates,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.65,
            method='LSTM'
        )
    
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"LSTM forecast failed: {e}")
        return None


def try_prophet_forecast(
    data: pd.DataFrame,
    horizon: int = 5
) -> Optional[ForecastResult]:
    """
    Attempt Prophet forecast if available.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
    
    Returns:
        ForecastResult or None
    """
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': data.index,
            'y': data['Close'].values
        })
        
        # Fit model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        
        # Forecast
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        # Extract predictions
        predictions = forecast['yhat'].tail(horizon).values
        lower = forecast['yhat_lower'].tail(horizon).values
        upper = forecast['yhat_upper'].tail(horizon).values
        future_dates = pd.DatetimeIndex(forecast['ds'].tail(horizon))
        
        return ForecastResult(
            predictions=predictions,
            dates=future_dates,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.7,
            method='Prophet'
        )
    
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"Prophet forecast failed: {e}")
        return None


def forecast_price(
    data: pd.DataFrame,
    horizon: int = 5,
    method: str = 'auto'
) -> ForecastResult:
    """
    Main forecasting function with automatic method selection.
    
    Args:
        data: OHLCV DataFrame
        horizon: Number of periods to forecast
        method: 'auto', 'ensemble', 'lstm', 'prophet', 'exponential', 'linear'
    
    Returns:
        ForecastResult
    """
    if len(data) < 30:
        return forecast_naive(data, horizon)
    
    if method == 'auto':
        # Try advanced methods first
        lstm_result = try_lstm_forecast(data, horizon)
        if lstm_result:
            return lstm_result
        
        prophet_result = try_prophet_forecast(data, horizon)
        if prophet_result:
            return prophet_result
        
        # Fall back to ensemble
        return forecast_ensemble(data, horizon)
    
    elif method == 'lstm':
        result = try_lstm_forecast(data, horizon)
        return result if result else forecast_ensemble(data, horizon)
    
    elif method == 'prophet':
        result = try_prophet_forecast(data, horizon)
        return result if result else forecast_ensemble(data, horizon)
    
    elif method == 'exponential':
        return forecast_exponential_smoothing(data, horizon)
    
    elif method == 'linear':
        return forecast_linear_regression(data, horizon)
    
    else:
        return forecast_ensemble(data, horizon)


def format_forecast_table(result: ForecastResult) -> pd.DataFrame:
    """Format forecast result as display table."""
    rows = []
    for i in range(len(result.predictions)):
        rows.append({
            'Date': result.dates[i].strftime('%Y-%m-%d'),
            'Forecast': f"${result.predictions[i]:.2f}",
            'Lower (95%)': f"${result.lower_bound[i]:.2f}",
            'Upper (95%)': f"${result.upper_bound[i]:.2f}"
        })
    
    return pd.DataFrame(rows)


def calculate_forecast_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        Dict with accuracy metrics
    """
    if len(actual) != len(predicted) or len(actual) == 0:
        return {}
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(actual))
    pred_direction = np.sign(np.diff(predicted))
    direction_accuracy = np.mean(actual_direction == pred_direction) if len(actual_direction) > 0 else 0
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'direction_accuracy': direction_accuracy
    }
