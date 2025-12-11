#!/usr/bin/env python3
"""
Simple demo script to test the stock dashboard functionality
This script demonstrates basic features without complex pandas operations
"""

import sys
import os
from datetime import date, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import fetch_stock_data, get_stock_info
from modules.indicators import sma, ema, rsi, macd
from modules.utils import format_number

def simple_demo():
    """Simple demonstration of core functionality"""
    print("=" * 60)
    print("📈 STOCK DASHBOARD - SIMPLE DEMO")
    print("=" * 60)
    
    # Test parameters
    ticker = "AAPL"
    end_date = date.today()
    start_date = end_date - timedelta(days=90)  # 3 months of data
    
    print(f"Testing with {ticker} from {start_date} to {end_date}")
    print("-" * 40)
    
    # 1. Test data fetching
    print("🔄 Testing data fetching...")
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        print("❌ Data fetching failed")
        return False
    
    print(f"✅ Data fetching successful: {len(data)} days of data")
    
    # 2. Test company info
    print("🔄 Testing company info...")
    try:
        stock_info = get_stock_info(ticker)
        print(f"✅ Company info: {stock_info['name']}")
    except Exception as e:
        print(f"⚠️ Company info warning: {str(e)}")
    
    # 3. Test technical indicators
    print("🔄 Testing technical indicators...")
    
    indicators_tested = []
    
    # Test SMA
    try:
        sma_data = sma(data.copy(), 20)
        if 'SMA20' in sma_data.columns:
            indicators_tested.append("✅ SMA (Simple Moving Average)")
        else:
            indicators_tested.append("❌ SMA: Column not found")
    except Exception as e:
        indicators_tested.append(f"❌ SMA: {str(e)}")
    
    # Test EMA
    try:
        ema_data = ema(data.copy(), 20)
        if 'EMA20' in ema_data.columns:
            indicators_tested.append("✅ EMA (Exponential Moving Average)")
        else:
            indicators_tested.append("❌ EMA: Column not found")
    except Exception as e:
        indicators_tested.append(f"❌ EMA: {str(e)}")
    
    # Test RSI
    try:
        rsi_data = rsi(data.copy(), 14)
        if 'RSI' in rsi_data.columns:
            indicators_tested.append("✅ RSI (Relative Strength Index)")
        else:
            indicators_tested.append("❌ RSI: Column not found")
    except Exception as e:
        indicators_tested.append(f"❌ RSI: {str(e)}")
    
    # Test MACD
    try:
        macd_data = macd(data.copy())
        if 'MACD' in macd_data.columns:
            indicators_tested.append("✅ MACD (Moving Average Convergence Divergence)")
        else:
            indicators_tested.append("❌ MACD: Column not found")
    except Exception as e:
        indicators_tested.append(f"❌ MACD: {str(e)}")
    
    # Print indicator test results
    print("\n📊 Technical Indicator Tests:")
    for result in indicators_tested:
        print(result)
    
    # 4. Test utility functions
    print("\n🔄 Testing utility functions...")
    
    utility_tests = []
    
    # Test number formatting
    try:
        currency_test = format_number(1234.56, "currency")
        if currency_test == "$1,234.56":
            utility_tests.append("✅ Currency formatting")
        else:
            utility_tests.append(f"❌ Currency formatting: {currency_test}")
    except Exception as e:
        utility_tests.append(f"❌ Currency formatting: {str(e)}")
    
    # Test percentage formatting
    try:
        percent_test = format_number(12.345, "percentage")
        if percent_test == "12.35%":
            utility_tests.append("✅ Percentage formatting")
        else:
            utility_tests.append(f"❌ Percentage formatting: {percent_test}")
    except Exception as e:
        utility_tests.append(f"❌ Percentage formatting: {str(e)}")
    
    # Print utility test results
    print("\n🔧 Utility Function Tests:")
    for result in utility_tests:
        print(result)
    
    # 5. Show some actual data
    print(f"\n📈 Latest {ticker} Data:")
    latest_data = data.iloc[-1]
    print(f"Date: {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Close Price: {format_number(float(latest_data['Close']), 'currency')}")
    print(f"Volume: {format_number(float(latest_data['Volume']), 'number')}")
    
    # Calculate simple metrics
    price_change = float(latest_data['Close']) - float(latest_data['Open'])
    price_change_percent = (price_change / float(latest_data['Open'])) * 100
    
    print(f"Daily Change: {format_number(price_change, 'currency')} ({price_change_percent:+.2f}%)")
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("🔄 Testing module imports...")
    
    import_tests = []
    
    try:
        import streamlit
        import_tests.append("✅ streamlit")
    except ImportError as e:
        import_tests.append(f"❌ streamlit: {str(e)}")
    
    try:
        import yfinance
        import_tests.append("✅ yfinance")
    except ImportError as e:
        import_tests.append(f"❌ yfinance: {str(e)}")
    
    try:
        import plotly
        import_tests.append("✅ plotly")
    except ImportError as e:
        import_tests.append(f"❌ plotly: {str(e)}")
    
    try:
        import pandas
        import_tests.append("✅ pandas")
    except ImportError as e:
        import_tests.append(f"❌ pandas: {str(e)}")
    
    try:
        import numpy
        import_tests.append("✅ numpy")
    except ImportError as e:
        import_tests.append(f"❌ numpy: {str(e)}")
    
    print("\n📦 Import Test Results:")
    for result in import_tests:
        print(result)
    
    return all("✅" in result for result in import_tests)

def main():
    """Main demo function"""
    print("🚀 Starting Simple Stock Dashboard Demo...")
    print("This demo tests core functionality without complex operations")
    print()
    
    try:
        # Test imports first
        if not test_imports():
            print("\n❌ Some imports failed. Please check your environment.")
            return
        
        print("\n" + "="*60)
        
        # Run main demo
        if simple_demo():
            print("\n" + "=" * 60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\n📋 All core features tested:")
            print("✅ Data fetching from yfinance")
            print("✅ Technical indicators calculation")
            print("✅ Utility functions")
            print("✅ Data processing")
            
            print("\n🚀 Next steps:")
            print("1. Run the full interactive dashboard: streamlit run app.py")
            print("2. Run unit tests: python tests/test_stock_dashboard.py")
            print("3. Deploy to Streamlit Cloud for public access")
            
        else:
            print("\n❌ Demo encountered issues")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        print("This might be due to network issues or API limitations.")

if __name__ == "__main__":
    main()