# Stock Dashboard Tests

This directory contains unit tests for the stock dashboard project.

## Running Tests

To run all tests:
```bash
python -m pytest tests/ -v
```

Or run the test file directly:
```bash
python tests/test_stock_dashboard.py
```

## Test Coverage

The tests cover:
- Data fetching functionality
- Technical indicators calculations
- Utility functions
- Integration workflows

## Test Categories

1. **TestDataFetcher**: Tests for data acquisition module
2. **TestIndicators**: Tests for technical indicators calculations
3. **TestUtils**: Tests for utility functions
4. **TestIntegration**: End-to-end integration tests

Note: Some tests require internet connection for real data fetching.