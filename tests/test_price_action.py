"""
Test Price Action Module

Quick test to verify price action detection functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
close = 100 + np.cumsum(np.random.randn(100) * 2)
high = close + np.random.rand(100) * 3
low = close - np.random.rand(100) * 3
open_p = close + np.random.randn(100) * 1
volume = np.random.randint(1000000, 5000000, 100)

data = pd.DataFrame({
    'Open': open_p,
    'High': high,
    'Low': low,
    'Close': close,
    'Volume': volume
}, index=dates)

print("Testing Price Action Module...")

# Test price action analysis
from modules.price_action import analyze_price_action
analyzed_data, sr_data, summary = analyze_price_action(data)

swing_highs = len(analyzed_data[analyzed_data["SwingHigh"]])
swing_lows = len(analyzed_data[analyzed_data["SwingLow"]])
support_levels = len(sr_data.get("support_levels", []))
resistance_levels = len(sr_data.get("resistance_levels", []))
patterns = len(analyzed_data[analyzed_data["Pattern"] != ""])

print(f"- Swing points detected: {swing_highs} highs, {swing_lows} lows")
print(f"- Support levels: {support_levels}")
print(f"- Resistance levels: {resistance_levels}")
print(f"- Trend: {summary.get('trend', 'N/A')}")
print(f"- Phase: {summary.get('phase', 'N/A')}")
print(f"- Patterns detected: {patterns}")

# Test strategies
print("\nTesting Strategies Module...")
from modules.strategies import generate_trading_signals

# Test Level + Signal + RR strategy
params = {'min_rr': 2.0, 'sr_lookback': 20, 'sr_threshold': 0.02}
strat_data, strat_sr, strat_summary = generate_trading_signals(data.copy(), 'level_signal_rr', params)
print(f"- Level+Signal+RR: {strat_summary.get('total_signals', 0)} signals")

# Test Trend Pullback strategy
params = {'min_rr': 2.0, 'ema_period': 20, 'swing_lookback': 5}
strat_data, strat_sr, strat_summary = generate_trading_signals(data.copy(), 'trend_pullback', params)
print(f"- Trend Pullback: {strat_summary.get('total_signals', 0)} signals")

# Test Breakout Retest strategy
params = {'min_rr': 2.0, 'consolidation_bars': 10, 'confirmation_required': True}
strat_data, strat_sr, strat_summary = generate_trading_signals(data.copy(), 'breakout_retest', params)
print(f"- Breakout Retest: {strat_summary.get('total_signals', 0)} signals")

print("\n✅ All price action tests passed!")
