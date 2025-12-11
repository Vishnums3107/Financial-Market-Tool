import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker, start, end):
    """
    Fetch stock data using yfinance with enhanced error handling.
    
    Args:
        ticker (str): Stock ticker symbol
        start (datetime/str): Start date
        end (datetime/str): End date
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        # Validate ticker format
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol")
        
        ticker = ticker.upper().strip()
        
        # Download data with progress indication
        with st.spinner(f"Fetching data for {ticker}..."):
            data = yf.download(ticker, start=start, end=end, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for {ticker}. Please check the ticker symbol.")
        
        # Flatten multi-level columns if present (yfinance returns MultiIndex columns)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Data validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns in data for {ticker}")
        
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        # Log successful fetch
        logger.info(f"Successfully fetched {len(data)} rows of data for {ticker}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

def get_stock_info(ticker):
    """
    Get basic stock information.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {str(e)}")
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 'N/A', 'pe_ratio': 'N/A'}

def validate_date_range(start_date, end_date):
    """
    Validate the date range for stock data fetching.
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
    
    Returns:
        bool: True if valid, False otherwise
    """
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return False
    
    if end_date > datetime.now().date():
        st.warning("End date is in the future. Using today's date.")
        return True
    
    if (end_date - start_date).days < 30:
        st.warning("Date range is less than 30 days. Some indicators may not be accurate.")
    
    return True
