import pandas as pd
import numpy as np

def sma(data, window=20):
    """Simple Moving Average"""
    data = data.copy()
    data[f"SMA{window}"] = data['Close'].rolling(window).mean()
    return data

def ema(data, window=20):
    """Exponential Moving Average"""
    data = data.copy()
    data[f"EMA{window}"] = data['Close'].ewm(span=window, adjust=False).mean()
    return data

def rsi(data, period=14):
    """Relative Strength Index"""
    data = data.copy()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def macd(data, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    data = data.copy()
    data['EMA12'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    return data

def bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    data = data.copy()
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    data['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    data['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    data['BB_Middle'] = rolling_mean
    return data

def stochastic_oscillator(data, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    data = data.copy()
    low_min = data['Low'].rolling(k_period).min()
    high_max = data['High'].rolling(k_period).max()
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(d_period).mean()
    return data

def williams_r(data, period=14):
    """Williams %R"""
    data = data.copy()
    high_max = data['High'].rolling(period).max()
    low_min = data['Low'].rolling(period).min()
    data['Williams_R'] = -100 * ((high_max - data['Close']) / (high_max - low_min))
    return data

def atr(data, period=14):
    """Average True Range"""
    data = data.copy()
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    data['ATR'] = true_range.rolling(period).mean()
    return data

def calculate_all_indicators(data, indicators_config):
    """Calculate multiple indicators based on configuration"""
    data_with_indicators = data.copy()
    
    for indicator, config in indicators_config.items():
        if indicator == 'SMA' and config['enabled']:
            for window in config['windows']:
                data_with_indicators = sma(data_with_indicators, window)
        elif indicator == 'EMA' and config['enabled']:
            for window in config['windows']:
                data_with_indicators = ema(data_with_indicators, window)
        elif indicator == 'RSI' and config['enabled']:
            data_with_indicators = rsi(data_with_indicators, config['period'])
        elif indicator == 'MACD' and config['enabled']:
            data_with_indicators = macd(data_with_indicators, config['fast'], config['slow'], config['signal'])
        elif indicator == 'Bollinger' and config['enabled']:
            data_with_indicators = bollinger_bands(data_with_indicators, config['window'], config['std'])
        elif indicator == 'Stochastic' and config['enabled']:
            data_with_indicators = stochastic_oscillator(data_with_indicators, config['k_period'], config['d_period'])
        elif indicator == 'Williams_R' and config['enabled']:
            data_with_indicators = williams_r(data_with_indicators, config['period'])
        elif indicator == 'ATR' and config['enabled']:
            data_with_indicators = atr(data_with_indicators, config['period'])
    
    return data_with_indicators
