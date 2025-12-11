"""
Price Action Analysis Module

Core price action detection functions for support/resistance,
trend structure, candlestick patterns, and market phases.

Universal concepts applicable across forex, stocks, indices, commodities, and crypto.
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def detect_swing_points(data, lookback=5):
    """
    Detect swing high and swing low points in price data.
    
    Args:
        data (pd.DataFrame): OHLCV data
        lookback (int): Number of bars to look back/forward for swing detection
    
    Returns:
        pd.DataFrame: Data with SwingHigh and SwingLow boolean columns
    """
    data = data.copy()
    
    # Get High and Low as numpy arrays
    highs = data['High'].values
    lows = data['Low'].values
    
    # Find local maxima and minima
    swing_high_idx = argrelextrema(highs, np.greater_equal, order=lookback)[0]
    swing_low_idx = argrelextrema(lows, np.less_equal, order=lookback)[0]
    
    # Create boolean columns
    data['SwingHigh'] = False
    data['SwingLow'] = False
    
    data.iloc[swing_high_idx, data.columns.get_loc('SwingHigh')] = True
    data.iloc[swing_low_idx, data.columns.get_loc('SwingLow')] = True
    
    # Store swing values
    data['SwingHighValue'] = np.nan
    data['SwingLowValue'] = np.nan
    
    data.loc[data['SwingHigh'], 'SwingHighValue'] = data.loc[data['SwingHigh'], 'High']
    data.loc[data['SwingLow'], 'SwingLowValue'] = data.loc[data['SwingLow'], 'Low']
    
    return data


def detect_support_resistance(data, lookback=20, threshold=0.02, min_touches=2):
    """
    Detect horizontal support and resistance zones using swing highs/lows.
    
    Args:
        data (pd.DataFrame): OHLCV data with swing points
        lookback (int): Lookback period for swing detection
        threshold (float): Percentage threshold for clustering levels
        min_touches (int): Minimum touches to confirm a level
    
    Returns:
        dict: Contains 'support_levels', 'resistance_levels', and 'zones'
    """
    data = data.copy()
    
    # Ensure swing points are detected
    if 'SwingHigh' not in data.columns:
        data = detect_swing_points(data, lookback=lookback // 4)
    
    # Get all swing highs and lows
    swing_highs = data.loc[data['SwingHigh'], 'High'].values
    swing_lows = data.loc[data['SwingLow'], 'Low'].values
    
    current_price = data['Close'].iloc[-1]
    
    def cluster_levels(levels, threshold_pct):
        """Cluster nearby price levels into zones."""
        if len(levels) == 0:
            return []
        
        levels = np.sort(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[0]) / current_cluster[0] <= threshold_pct:
                current_cluster.append(level)
            else:
                if len(current_cluster) >= min_touches:
                    clusters.append({
                        'level': np.mean(current_cluster),
                        'strength': len(current_cluster),
                        'range': (min(current_cluster), max(current_cluster))
                    })
                current_cluster = [level]
        
        # Don't forget the last cluster
        if len(current_cluster) >= min_touches:
            clusters.append({
                'level': np.mean(current_cluster),
                'strength': len(current_cluster),
                'range': (min(current_cluster), max(current_cluster))
            })
        
        return clusters
    
    # Cluster all swing points
    all_levels = np.concatenate([swing_highs, swing_lows])
    clustered = cluster_levels(all_levels, threshold)
    
    # Separate into support and resistance based on current price
    support_levels = [z for z in clustered if z['level'] < current_price]
    resistance_levels = [z for z in clustered if z['level'] >= current_price]
    
    # Sort by proximity to current price
    support_levels = sorted(support_levels, key=lambda x: -x['level'])
    resistance_levels = sorted(resistance_levels, key=lambda x: x['level'])
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'all_zones': clustered,
        'current_price': current_price
    }


def identify_trend_structure(data, swing_lookback=5):
    """
    Identify market structure using Higher Highs/Higher Lows and Lower Highs/Lower Lows.
    
    Args:
        data (pd.DataFrame): OHLCV data
        swing_lookback (int): Lookback for swing point detection
    
    Returns:
        pd.DataFrame: Data with trend structure columns
    """
    data = data.copy()
    
    # Detect swing points if not already done
    if 'SwingHigh' not in data.columns:
        data = detect_swing_points(data, lookback=swing_lookback)
    
    # Initialize structure columns
    data['TrendStructure'] = 'Neutral'
    data['SwingType'] = ''
    
    # Get swing point indices and values
    swing_high_indices = data[data['SwingHigh']].index.tolist()
    swing_low_indices = data[data['SwingLow']].index.tolist()
    
    # Classify swing highs
    last_swing_high = None
    for idx in swing_high_indices:
        current_high = data.loc[idx, 'High']
        if last_swing_high is not None:
            if current_high > last_swing_high:
                data.loc[idx, 'SwingType'] = 'HH'  # Higher High
            else:
                data.loc[idx, 'SwingType'] = 'LH'  # Lower High
        last_swing_high = current_high
    
    # Classify swing lows
    last_swing_low = None
    for idx in swing_low_indices:
        current_low = data.loc[idx, 'Low']
        if last_swing_low is not None:
            if current_low > last_swing_low:
                data.loc[idx, 'SwingType'] = 'HL'  # Higher Low
            else:
                data.loc[idx, 'SwingType'] = 'LL'  # Lower Low
        last_swing_low = current_low
    
    # Determine overall trend structure for each bar
    # Forward fill the last known structure
    recent_swings = []
    for i, row in data.iterrows():
        if row['SwingType'] in ['HH', 'HL', 'LH', 'LL']:
            recent_swings.append(row['SwingType'])
            if len(recent_swings) > 4:
                recent_swings.pop(0)
        
        # Determine trend from recent swings
        if len(recent_swings) >= 2:
            hh_count = recent_swings.count('HH')
            hl_count = recent_swings.count('HL')
            lh_count = recent_swings.count('LH')
            ll_count = recent_swings.count('LL')
            
            if (hh_count + hl_count) > (lh_count + ll_count):
                data.loc[i, 'TrendStructure'] = 'Uptrend'
            elif (lh_count + ll_count) > (hh_count + hl_count):
                data.loc[i, 'TrendStructure'] = 'Downtrend'
            else:
                data.loc[i, 'TrendStructure'] = 'Ranging'
    
    return data


def classify_market_phase(data, atr_period=14, lookback=20):
    """
    Classify market into trending or ranging phases.
    
    Args:
        data (pd.DataFrame): OHLCV data
        atr_period (int): Period for ATR calculation
        lookback (int): Lookback for phase classification
    
    Returns:
        pd.DataFrame: Data with MarketPhase column
    """
    data = data.copy()
    
    # Calculate ATR if not present
    if 'ATR' not in data.columns:
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(atr_period).mean()
    
    # Calculate price range and directional movement over lookback
    data['PriceRange'] = data['High'].rolling(lookback).max() - data['Low'].rolling(lookback).min()
    data['DirectionalMove'] = np.abs(data['Close'] - data['Close'].shift(lookback))
    
    # Efficiency Ratio: how much of the range was used for directional movement
    data['EfficiencyRatio'] = data['DirectionalMove'] / data['PriceRange']
    data['EfficiencyRatio'] = data['EfficiencyRatio'].fillna(0)
    
    # Classify phase
    data['MarketPhase'] = 'Unknown'
    data.loc[data['EfficiencyRatio'] > 0.5, 'MarketPhase'] = 'Trending'
    data.loc[data['EfficiencyRatio'] <= 0.5, 'MarketPhase'] = 'Ranging'
    
    # Also consider volatility compression for ranging
    avg_atr = data['ATR'].rolling(lookback * 2).mean()
    data['VolatilityRatio'] = data['ATR'] / avg_atr
    data.loc[data['VolatilityRatio'] < 0.7, 'MarketPhase'] = 'Consolidating'
    
    return data


def detect_candlestick_patterns(data, body_threshold=0.3, wick_threshold=2.0):
    """
    Detect key price action candlestick patterns.
    
    Args:
        data (pd.DataFrame): OHLCV data
        body_threshold (float): Maximum body/range ratio for pin bars
        wick_threshold (float): Minimum wick/body ratio for pin bars
    
    Returns:
        pd.DataFrame: Data with pattern detection columns
    """
    data = data.copy()
    
    # Calculate candle components
    data['Body'] = np.abs(data['Close'] - data['Open'])
    data['Range'] = data['High'] - data['Low']
    data['UpperWick'] = data['High'] - np.maximum(data['Close'], data['Open'])
    data['LowerWick'] = np.minimum(data['Close'], data['Open']) - data['Low']
    data['IsBullish'] = data['Close'] > data['Open']
    
    # Body to range ratio (small body = potential reversal candle)
    data['BodyRatio'] = data['Body'] / data['Range'].replace(0, np.nan)
    data['BodyRatio'] = data['BodyRatio'].fillna(0)
    
    # Initialize pattern columns
    data['Pattern'] = ''
    data['PatternDirection'] = ''
    
    # Pin Bar (Hammer/Shooting Star)
    # Long lower wick = bullish pin bar (hammer)
    bullish_pin = (
        (data['LowerWick'] > data['Body'] * wick_threshold) &
        (data['UpperWick'] < data['Body']) &
        (data['BodyRatio'] < body_threshold * 2)
    )
    
    # Long upper wick = bearish pin bar (shooting star)
    bearish_pin = (
        (data['UpperWick'] > data['Body'] * wick_threshold) &
        (data['LowerWick'] < data['Body']) &
        (data['BodyRatio'] < body_threshold * 2)
    )
    
    data.loc[bullish_pin, 'Pattern'] = 'PinBar'
    data.loc[bullish_pin, 'PatternDirection'] = 'Bullish'
    data.loc[bearish_pin, 'Pattern'] = 'PinBar'
    data.loc[bearish_pin, 'PatternDirection'] = 'Bearish'
    
    # Engulfing Pattern
    prev_body = data['Body'].shift(1)
    prev_open = data['Open'].shift(1)
    prev_close = data['Close'].shift(1)
    prev_bullish = data['IsBullish'].shift(1).fillna(False)
    
    # Bullish engulfing: previous bearish, current bullish, current body engulfs previous
    bullish_engulfing = (
        (prev_bullish == False) & 
        (data['IsBullish'] == True) &
        (data['Open'] <= prev_close) &
        (data['Close'] >= prev_open) &
        (data['Body'] > prev_body)
    ).fillna(False)
    
    # Bearish engulfing: previous bullish, current bearish, current body engulfs previous
    bearish_engulfing = (
        (prev_bullish == True) &
        (data['IsBullish'] == False) &
        (data['Open'] >= prev_close) &
        (data['Close'] <= prev_open) &
        (data['Body'] > prev_body)
    ).fillna(False)
    
    data.loc[bullish_engulfing, 'Pattern'] = 'Engulfing'
    data.loc[bullish_engulfing, 'PatternDirection'] = 'Bullish'
    data.loc[bearish_engulfing, 'Pattern'] = 'Engulfing'
    data.loc[bearish_engulfing, 'PatternDirection'] = 'Bearish'
    
    # Inside Bar (range within previous bar)
    prev_high = data['High'].shift(1)
    prev_low = data['Low'].shift(1)
    
    inside_bar = (
        (data['High'] <= prev_high) &
        (data['Low'] >= prev_low)
    )
    
    # Only mark as inside bar if no other pattern detected
    data.loc[inside_bar & (data['Pattern'] == ''), 'Pattern'] = 'InsideBar'
    data.loc[inside_bar & (data['PatternDirection'] == ''), 'PatternDirection'] = 'Neutral'
    
    # Large Impulse Candle (strong momentum)
    avg_range = data['Range'].rolling(20).mean()
    large_body = (
        (data['Body'] > avg_range * 1.5) &
        (data['BodyRatio'] > 0.6)
    )
    
    data.loc[large_body & (data['Pattern'] == '') & data['IsBullish'], 'Pattern'] = 'ImpulseCandle'
    data.loc[large_body & (data['Pattern'] == 'ImpulseCandle') & data['IsBullish'], 'PatternDirection'] = 'Bullish'
    data.loc[large_body & (data['Pattern'] == '') & ~data['IsBullish'], 'Pattern'] = 'ImpulseCandle'
    data.loc[large_body & (data['Pattern'] == 'ImpulseCandle') & ~data['IsBullish'], 'PatternDirection'] = 'Bearish'
    
    # Doji (very small body)
    doji = data['BodyRatio'] < 0.1
    data.loc[doji & (data['Pattern'] == ''), 'Pattern'] = 'Doji'
    data.loc[doji & (data['PatternDirection'] == ''), 'PatternDirection'] = 'Neutral'
    
    return data


def detect_breakouts_retests(data, sr_data, lookback=5, confirmation_bars=2):
    """
    Detect breakouts beyond S/R levels and subsequent retests.
    
    Args:
        data (pd.DataFrame): OHLCV data
        sr_data (dict): Support/resistance data from detect_support_resistance()
        lookback (int): Bars to look back for recent breakouts
        confirmation_bars (int): Bars to confirm a retest
    
    Returns:
        pd.DataFrame: Data with breakout/retest detection columns
    """
    data = data.copy()
    
    # Initialize columns
    data['BreakoutType'] = ''
    data['BreakoutLevel'] = np.nan
    data['IsRetest'] = False
    data['RetestLevel'] = np.nan
    
    all_zones = sr_data.get('all_zones', [])
    
    if not all_zones:
        return data
    
    # Check each bar for breakouts and retests
    for i in range(lookback, len(data)):
        current_close = data['Close'].iloc[i]
        prev_close = data['Close'].iloc[i - 1]
        current_low = data['Low'].iloc[i]
        current_high = data['High'].iloc[i]
        
        for zone in all_zones:
            level = zone['level']
            zone_range = zone['range']
            zone_low, zone_high = zone_range
            
            # Check for upward breakout
            if prev_close < zone_high and current_close > zone_high:
                data.iloc[i, data.columns.get_loc('BreakoutType')] = 'Bullish'
                data.iloc[i, data.columns.get_loc('BreakoutLevel')] = level
            
            # Check for downward breakout
            elif prev_close > zone_low and current_close < zone_low:
                data.iloc[i, data.columns.get_loc('BreakoutType')] = 'Bearish'
                data.iloc[i, data.columns.get_loc('BreakoutLevel')] = level
            
            # Check for retest (price comes back to the level after breakout)
            # Look for recent breakouts in the lookback window
            recent_breakouts = data.iloc[max(0, i-lookback):i]
            
            for j, bo in recent_breakouts.iterrows():
                if bo['BreakoutType'] == 'Bullish' and pd.notna(bo['BreakoutLevel']):
                    # After bullish breakout, price should retest from above
                    if current_low <= bo['BreakoutLevel'] * 1.02 and current_close > bo['BreakoutLevel']:
                        data.iloc[i, data.columns.get_loc('IsRetest')] = True
                        data.iloc[i, data.columns.get_loc('RetestLevel')] = bo['BreakoutLevel']
                
                elif bo['BreakoutType'] == 'Bearish' and pd.notna(bo['BreakoutLevel']):
                    # After bearish breakout, price should retest from below
                    if current_high >= bo['BreakoutLevel'] * 0.98 and current_close < bo['BreakoutLevel']:
                        data.iloc[i, data.columns.get_loc('IsRetest')] = True
                        data.iloc[i, data.columns.get_loc('RetestLevel')] = bo['BreakoutLevel']
    
    return data


def get_price_action_summary(data):
    """
    Generate a summary of price action analysis.
    
    Args:
        data (pd.DataFrame): Data with all price action columns
    
    Returns:
        dict: Summary of current price action state
    """
    summary = {}
    
    # Current trend structure
    if 'TrendStructure' in data.columns:
        summary['trend'] = data['TrendStructure'].iloc[-1]
    
    # Current market phase
    if 'MarketPhase' in data.columns:
        summary['phase'] = data['MarketPhase'].iloc[-1]
    
    # Recent patterns
    if 'Pattern' in data.columns:
        recent_patterns = data[data['Pattern'] != ''].tail(5)
        summary['recent_patterns'] = recent_patterns[['Pattern', 'PatternDirection']].to_dict('records')
    
    # Recent breakouts/retests
    if 'BreakoutType' in data.columns:
        recent_breakouts = data[data['BreakoutType'] != ''].tail(3)
        summary['recent_breakouts'] = recent_breakouts[['BreakoutType', 'BreakoutLevel']].to_dict('records')
    
    if 'IsRetest' in data.columns:
        recent_retests = data[data['IsRetest']].tail(3)
        summary['recent_retests'] = recent_retests[['RetestLevel']].to_dict('records')
    
    return summary


def analyze_price_action(data, swing_lookback=5, sr_lookback=20, sr_threshold=0.02):
    """
    Perform complete price action analysis on data.
    
    Args:
        data (pd.DataFrame): OHLCV data
        swing_lookback (int): Lookback for swing point detection
        sr_lookback (int): Lookback for S/R detection
        sr_threshold (float): Threshold for S/R level clustering
    
    Returns:
        tuple: (analyzed_data, sr_data, summary)
    """
    # Step 1: Detect swing points
    data = detect_swing_points(data, lookback=swing_lookback)
    
    # Step 2: Identify support/resistance
    sr_data = detect_support_resistance(data, lookback=sr_lookback, threshold=sr_threshold)
    
    # Step 3: Identify trend structure
    data = identify_trend_structure(data, swing_lookback=swing_lookback)
    
    # Step 4: Classify market phase
    data = classify_market_phase(data)
    
    # Step 5: Detect candlestick patterns
    data = detect_candlestick_patterns(data)
    
    # Step 6: Detect breakouts and retests
    data = detect_breakouts_retests(data, sr_data)
    
    # Step 7: Generate summary
    summary = get_price_action_summary(data)
    
    return data, sr_data, summary
