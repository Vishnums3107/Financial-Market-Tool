#!/usr/bin/env python3
"""
Demo script to test the stock dashboard functionality
This script demonstrates the core features without the Streamlit UI
"""

import sys
import os
from datetime import date, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import fetch_stock_data, get_stock_info
from modules.indicators import sma, ema, rsi, macd, bollinger_bands
from modules.utils import format_number, calculate_portfolio_metrics

def demo_single_stock_analysis():
    """Demonstrate single stock analysis"""
    print("=" * 60)
    print("📈 STOCK DASHBOARD DEMO - SINGLE STOCK ANALYSIS")
    print("=" * 60)
    
    # Test parameters
    ticker = "AAPL"
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    print(f"Analyzing {ticker} from {start_date} to {end_date}")
    print("-" * 40)
    
    # Fetch stock data
    print("🔄 Fetching stock data...")
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        print("❌ Failed to fetch data")
        return
    
    print(f"✅ Successfully fetched {len(data)} days of data")
    
    # Get stock info
    print("🔄 Fetching company information...")
    stock_info = get_stock_info(ticker)
    print(f"Company: {stock_info['name']}")
    print(f"Sector: {stock_info['sector']}")
    print(f"Market Cap: {format_number(stock_info['market_cap'], 'market_cap')}")
    
    # Calculate technical indicators
    print("\n🔄 Calculating technical indicators...")
    data_with_indicators = data.copy()
    data_with_indicators = sma(data_with_indicators, 20)
    data_with_indicators = sma(data_with_indicators, 50)
    data_with_indicators = ema(data_with_indicators, 20)
    data_with_indicators = rsi(data_with_indicators, 14)
    data_with_indicators = macd(data_with_indicators)
    data_with_indicators = bollinger_bands(data_with_indicators)
    
    # Display current metrics
    current_data = data_with_indicators.iloc[-1]
    print(f"\n📊 Current Metrics for {ticker}:")
    print(f"Price: {format_number(float(current_data['Close']), 'currency')}")
    print(f"20-day SMA: {format_number(float(current_data['SMA20']), 'currency')}")
    print(f"50-day SMA: {format_number(float(current_data['SMA50']), 'currency')}")
    print(f"RSI: {float(current_data['RSI']):.2f}")
    
    # Technical signals
    print("\n🎯 Technical Signals:")
    rsi_value = float(current_data['RSI'])
    close_price = float(current_data['Close'])
    sma50_value = float(current_data['SMA50'])
    macd_value = float(current_data['MACD'])
    signal_value = float(current_data['Signal'])
    
    if rsi_value > 70:
        print("🔴 RSI: Overbought")
    elif rsi_value < 30:
        print("🟢 RSI: Oversold")
    else:
        print("🔵 RSI: Neutral")
    
    if close_price > sma50_value:
        print("🟢 Price above 50-day SMA (Bullish)")
    else:
        print("🔴 Price below 50-day SMA (Bearish)")
    
    if macd_value > signal_value:
        print("🟢 MACD: Bullish crossover")
    else:
        print("🔴 MACD: Bearish crossover")
    
    print("\n✅ Single stock analysis completed!")

def demo_portfolio_analysis():
    """Demonstrate portfolio analysis"""
    print("\n" + "=" * 60)
    print("📊 STOCK DASHBOARD DEMO - PORTFOLIO ANALYSIS")
    print("=" * 60)
    
    # Test portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    end_date = date.today()
    start_date = end_date - timedelta(days=180)  # 6 months of data
    
    print(f"Analyzing portfolio: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print("-" * 40)
    
    # Fetch data for all stocks
    data_dict = {}
    for ticker in tickers:
        print(f"🔄 Fetching data for {ticker}...")
        data = fetch_stock_data(ticker, start_date, end_date)
        if not data.empty:
            data_dict[ticker] = data
            print(f"✅ {ticker}: {len(data)} days of data")
        else:
            print(f"❌ {ticker}: Failed to fetch data")
    
    if len(data_dict) < 2:
        print("❌ Insufficient data for portfolio analysis")
        return
    
    # Calculate portfolio metrics
    print("\n🔄 Calculating portfolio metrics...")
    portfolio_metrics = calculate_portfolio_metrics(data_dict)
    
    print("\n📈 Portfolio Performance:")
    for metric, value in portfolio_metrics.items():
        print(f"{metric}: {value}")
    
    # Individual stock performance
    print("\n📊 Individual Stock Performance:")
    for ticker, data in data_dict.items():
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        
        print(f"{ticker}: {format_number(start_price, 'currency')} → "
              f"{format_number(end_price, 'currency')} "
              f"({total_return:+.2f}%)")
    
    print("\n✅ Portfolio analysis completed!")

def demo_indicator_calculations():
    """Demonstrate technical indicator calculations"""
    print("\n" + "=" * 60)
    print("🔧 STOCK DASHBOARD DEMO - INDICATOR CALCULATIONS")
    print("=" * 60)
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    print("🔄 Creating sample data for indicator testing...")
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic stock price data
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    sample_data = pd.DataFrame({
        'Open': prices + np.random.randn(len(dates)) * 0.1,
        'High': prices + np.abs(np.random.randn(len(dates)) * 0.5),
        'Low': prices - np.abs(np.random.randn(len(dates)) * 0.5),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    print(f"✅ Created {len(sample_data)} days of sample data")
    
    # Test each indicator
    indicators_tested = []
    
    print("\n🔄 Testing technical indicators...")
    
    # SMA
    try:
        result = sma(sample_data.copy(), 20)
        indicators_tested.append("✅ SMA (Simple Moving Average)")
    except Exception as e:
        indicators_tested.append(f"❌ SMA: {str(e)}")
    
    # EMA
    try:
        result = ema(sample_data.copy(), 20)
        indicators_tested.append("✅ EMA (Exponential Moving Average)")
    except Exception as e:
        indicators_tested.append(f"❌ EMA: {str(e)}")
    
    # RSI
    try:
        result = rsi(sample_data.copy(), 14)
        indicators_tested.append("✅ RSI (Relative Strength Index)")
    except Exception as e:
        indicators_tested.append(f"❌ RSI: {str(e)}")
    
    # MACD
    try:
        result = macd(sample_data.copy())
        indicators_tested.append("✅ MACD (Moving Average Convergence Divergence)")
    except Exception as e:
        indicators_tested.append(f"❌ MACD: {str(e)}")
    
    # Bollinger Bands
    try:
        result = bollinger_bands(sample_data.copy())
        indicators_tested.append("✅ Bollinger Bands")
    except Exception as e:
        indicators_tested.append(f"❌ Bollinger Bands: {str(e)}")
    
    print("\n📊 Indicator Test Results:")
    for result in indicators_tested:
        print(result)
    
    print("\n✅ Indicator calculations completed!")

def main():
    """Main demo function"""
    print("🚀 Starting Stock Dashboard Demo...")
    print("This demo showcases the core functionality of the dashboard")
    
    try:
        # Run demos
        demo_single_stock_analysis()
        demo_portfolio_analysis()
        demo_indicator_calculations()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("✅ Stock data fetching")
        print("✅ Technical indicators calculation")
        print("✅ Portfolio analysis")
        print("✅ Company information retrieval")
        print("✅ Performance metrics")
        
        print("\n🚀 To run the full interactive dashboard:")
        print("   streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()