import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_fetcher import fetch_stock_data, get_stock_info, validate_date_range
from modules.indicators import sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator
from modules.utils import format_number, validate_tickers, calculate_max_drawdown

class TestDataFetcher(unittest.TestCase):
    """Test cases for data fetcher module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_ticker = "AAPL"
        self.invalid_ticker = "INVALID123"
        self.start_date = date(2023, 1, 1)
        self.end_date = date(2023, 12, 31)
    
    def test_fetch_valid_ticker(self):
        """Test fetching data for a valid ticker"""
        # Note: This test requires internet connection
        # In a production environment, you might want to mock this
        data = fetch_stock_data(self.valid_ticker, self.start_date, self.end_date)
        
        # Check if data is not empty and has required columns
        self.assertFalse(data.empty)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
    
    def test_validate_date_range(self):
        """Test date range validation"""
        # Valid date range
        self.assertTrue(validate_date_range(self.start_date, self.end_date))
        
        # Invalid date range (start after end)
        self.assertFalse(validate_date_range(self.end_date, self.start_date))

class TestIndicators(unittest.TestCase):
    """Test cases for technical indicators"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock price data
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Open': prices + np.random.randn(len(dates)) * 0.1,
            'High': prices + np.abs(np.random.randn(len(dates)) * 0.5),
            'Low': prices - np.abs(np.random.randn(len(dates)) * 0.5),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        result = sma(self.sample_data.copy(), window=5)
        
        # Check if SMA column was added
        self.assertIn('SMA5', result.columns)
        
        # Check if SMA values are reasonable (not NaN after sufficient data)
        self.assertFalse(result['SMA5'].iloc[10:].isna().any())
        
        # Verify SMA calculation for a specific point
        manual_sma = self.sample_data['Close'].iloc[4:9].mean()
        calculated_sma = result['SMA5'].iloc[8]
        self.assertAlmostEqual(manual_sma, calculated_sma, places=5)
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        result = ema(self.sample_data.copy(), window=5)
        
        # Check if EMA column was added
        self.assertIn('EMA5', result.columns)
        
        # Check if EMA values are not NaN after first value
        self.assertFalse(result['EMA5'].iloc[1:].isna().any())
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        result = rsi(self.sample_data.copy(), period=14)
        
        # Check if RSI column was added
        self.assertIn('RSI', result.columns)
        
        # Check RSI bounds (should be between 0 and 100)
        rsi_values = result['RSI'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        result = macd(self.sample_data.copy())
        
        # Check if all MACD columns were added
        expected_columns = ['EMA12', 'EMA26', 'MACD', 'Signal', 'Histogram']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Verify histogram calculation
        macd_values = result['MACD'].dropna()
        signal_values = result['Signal'].dropna()
        histogram_values = result['Histogram'].dropna()
        
        # Check if histogram equals MACD minus Signal
        common_index = macd_values.index.intersection(signal_values.index).intersection(histogram_values.index)
        for idx in common_index[-10:]:  # Check last 10 values
            expected_histogram = macd_values[idx] - signal_values[idx]
            actual_histogram = histogram_values[idx]
            self.assertAlmostEqual(expected_histogram, actual_histogram, places=5)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        result = bollinger_bands(self.sample_data.copy(), window=20, num_std=2)
        
        # Check if Bollinger Bands columns were added
        expected_columns = ['BB_Upper', 'BB_Lower', 'BB_Middle']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check logical relationship: Lower < Middle < Upper
        bb_data = result[['BB_Lower', 'BB_Middle', 'BB_Upper']].dropna()
        for idx in bb_data.index[-10:]:  # Check last 10 values
            self.assertLess(bb_data.loc[idx, 'BB_Lower'], bb_data.loc[idx, 'BB_Middle'])
            self.assertLess(bb_data.loc[idx, 'BB_Middle'], bb_data.loc[idx, 'BB_Upper'])

class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_format_number(self):
        """Test number formatting"""
        # Test currency formatting
        self.assertEqual(format_number(1234.56, "currency"), "$1,234.56")
        
        # Test percentage formatting
        self.assertEqual(format_number(12.345, "percentage"), "12.35%")
        
        # Test market cap formatting
        self.assertEqual(format_number(1500000000, "market_cap"), "$1.50B")
        self.assertEqual(format_number(1500000000000, "market_cap"), "$1.50T")
        
        # Test N/A handling
        self.assertEqual(format_number('N/A', "currency"), "N/A")
    
    def test_validate_tickers(self):
        """Test ticker validation"""
        # Test valid tickers
        tickers = ["AAPL", "msft", " GOOGL ", "AMZN"]
        result = validate_tickers(tickers)
        expected = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        self.assertEqual(sorted(result), sorted(expected))
        
        # Test empty and invalid tickers
        tickers = ["", "  ", "AAPL", ""]
        result = validate_tickers(tickers)
        self.assertEqual(result, ["AAPL"])
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create a sample returns series with known drawdown
        returns = pd.Series([0.1, -0.05, -0.1, 0.05, 0.2, -0.15, 0.1])
        
        max_dd = calculate_max_drawdown(returns)
        
        # Max drawdown should be negative
        self.assertLess(max_dd, 0)
        
        # Test with positive returns only
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.03])
        max_dd_positive = calculate_max_drawdown(positive_returns)
        
        # Should be zero or very close to zero for only positive returns
        self.assertGreaterEqual(max_dd_positive, -0.001)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data for integration testing
        dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Open': prices + np.random.randn(len(dates)) * 0.1,
            'High': prices + np.abs(np.random.randn(len(dates)) * 0.5),
            'Low': prices - np.abs(np.random.randn(len(dates)) * 0.5),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_multiple_indicators_workflow(self):
        """Test applying multiple indicators in sequence"""
        data = self.sample_data.copy()
        
        # Apply multiple indicators
        data = sma(data, 20)
        data = ema(data, 12)
        data = rsi(data, 14)
        data = macd(data)
        data = bollinger_bands(data)
        
        # Check that all indicators were applied
        expected_columns = [
            'SMA20', 'EMA12', 'RSI', 'MACD', 'Signal', 'Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Middle'
        ]
        
        for col in expected_columns:
            self.assertIn(col, data.columns)
        
        # Verify data integrity
        self.assertEqual(len(data), len(self.sample_data))
        self.assertTrue(data.index.equals(self.sample_data.index))

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataFetcher))
    test_suite.addTest(unittest.makeSuite(TestIndicators))
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")