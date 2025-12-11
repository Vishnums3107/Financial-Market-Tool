# Utility functions for the stock dashboard project

import pandas as pd
import streamlit as st
import io
from datetime import datetime, timedelta
import base64

def export_data_to_csv(data, filename_prefix="stock_data"):
    """
    Export data to CSV and provide download link.
    
    Args:
        data (pd.DataFrame): Data to export
        filename_prefix (str): Prefix for filename
    
    Returns:
        str: Download link HTML
    """
    if data.empty:
        return None
    
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=True)
    csv_string = csv_buffer.getvalue()
    
    # Create download button
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    
    return href

def format_number(value, format_type="currency"):
    """
    Format numbers for display.
    
    Args:
        value: Number to format
        format_type (str): Type of formatting
    
    Returns:
        str: Formatted number
    """
    if pd.isna(value) or value == 'N/A':
        return 'N/A'
    
    try:
        if format_type == "currency":
            return f"${value:,.2f}"
        elif format_type == "percentage":
            return f"{value:.2f}%"
        elif format_type == "number":
            return f"{value:,.0f}"
        elif format_type == "market_cap":
            if value >= 1e12:
                return f"${value/1e12:.2f}T"
            elif value >= 1e9:
                return f"${value/1e9:.2f}B"
            elif value >= 1e6:
                return f"${value/1e6:.2f}M"
            else:
                return f"${value:,.0f}"
        else:
            return str(value)
    except:
        return str(value)

def calculate_portfolio_metrics(data_dict, weights=None):
    """
    Calculate portfolio-level metrics.
    
    Args:
        data_dict (dict): Dictionary of stock data
        weights (dict): Portfolio weights for each stock
    
    Returns:
        dict: Portfolio metrics
    """
    if not data_dict:
        return {}
    
    if weights is None:
        # Equal weights
        weights = {ticker: 1.0/len(data_dict) for ticker in data_dict.keys()}
    
    # Calculate portfolio returns
    portfolio_returns = []
    
    for ticker, weight in weights.items():
        if ticker in data_dict and not data_dict[ticker].empty:
            returns = data_dict[ticker]['Close'].pct_change().dropna()
            weighted_returns = returns * weight
            portfolio_returns.append(weighted_returns)
    
    if not portfolio_returns:
        return {}
    
    # Combine portfolio returns
    portfolio_return_series = sum(portfolio_returns)
    
    # Calculate metrics
    total_return = (1 + portfolio_return_series).prod() - 1
    annualized_return = (1 + portfolio_return_series.mean()) ** 252 - 1
    volatility = portfolio_return_series.std() * (252 ** 0.5)
    
    # Convert to float values for comparison - handle Series properly
    try:
        volatility_value = float(volatility) if pd.notna(volatility) and volatility != 0 else 0
    except (TypeError, ValueError):
        volatility_value = 0
    
    try:
        annualized_return_value = float(annualized_return)
    except (TypeError, ValueError):
        annualized_return_value = 0
    
    sharpe_ratio = annualized_return_value / volatility_value if volatility_value != 0 else 0
    
    max_drawdown = calculate_max_drawdown(portfolio_return_series)
    
    # Convert all Series to float values properly
    try:
        total_return_value = float(total_return.iloc[0]) if hasattr(total_return, 'iloc') else float(total_return)
    except (TypeError, ValueError, AttributeError):
        total_return_value = 0
    
    try:
        max_drawdown_value = float(max_drawdown.iloc[0]) if hasattr(max_drawdown, 'iloc') else float(max_drawdown)
    except (TypeError, ValueError, AttributeError):
        max_drawdown_value = 0
    
    return {
        'Total Return': f"{total_return_value:.2%}",
        'Annualized Return': f"{annualized_return_value:.2%}",
        'Volatility': f"{volatility_value:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown_value:.2%}"
    }

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from returns series.
    
    Args:
        returns (pd.Series): Returns series
    
    Returns:
        float: Maximum drawdown
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def get_popular_tickers():
    """
    Get list of popular stock tickers.
    
    Returns:
        dict: Categories of popular tickers
    """
    return {
        'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
        'Consumer': ['KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
        'ETFs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'IVV']
    }

def validate_tickers(tickers):
    """
    Validate ticker symbols.
    
    Args:
        tickers (list): List of ticker symbols
    
    Returns:
        list: Valid ticker symbols
    """
    valid_tickers = []
    for ticker in tickers:
        if isinstance(ticker, str) and len(ticker.strip()) > 0:
            valid_tickers.append(ticker.upper().strip())
    
    return list(set(valid_tickers))  # Remove duplicates

def create_indicator_config():
    """
    Create default indicator configuration.
    
    Returns:
        dict: Indicator configuration
    """
    return {
        'SMA': {
            'enabled': False,
            'windows': [20, 50, 200]
        },
        'EMA': {
            'enabled': False,
            'windows': [12, 26]
        },
        'RSI': {
            'enabled': False,
            'period': 14
        },
        'MACD': {
            'enabled': False,
            'fast': 12,
            'slow': 26,
            'signal': 9
        },
        'Bollinger': {
            'enabled': False,
            'window': 20,
            'std': 2
        },
        'Stochastic': {
            'enabled': False,
            'k_period': 14,
            'd_period': 3
        },
        'Williams_R': {
            'enabled': False,
            'period': 14
        },
        'ATR': {
            'enabled': False,
            'period': 14
        }
    }
