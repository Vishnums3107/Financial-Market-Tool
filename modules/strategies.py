"""
Trading Strategies Module

Rule-based trading strategies implementing universal price action concepts.
Each strategy generates actionable trade signals with entry, stop, and target levels.
"""

import pandas as pd
import numpy as np
from modules.price_action import (
    detect_swing_points,
    detect_support_resistance,
    identify_trend_structure,
    detect_candlestick_patterns,
    detect_breakouts_retests,
    classify_market_phase
)


def calculate_risk_reward(entry, stop, target):
    """
    Calculate risk/reward ratio for a trade setup.
    
    Args:
        entry (float): Entry price
        stop (float): Stop loss price
        target (float): Take profit price
    
    Returns:
        float: Risk/reward ratio (target/risk)
    """
    if entry == stop:
        return 0
    
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    if risk == 0:
        return 0
    
    return reward / risk


def level_signal_rr_strategy(data, min_rr=2.0, sr_lookback=20, sr_threshold=0.02):
    """
    Level + Signal + RR Strategy
    
    Trade price action signals at horizontal support/resistance levels
    with fixed risk/reward targets.
    
    Rules:
    1. Identify clear S/R zone from recent swing highs/lows
    2. Wait for strong candlestick signal at level (pin bar, engulfing)
    3. Enter in direction of rejection
    4. Stop loss beyond the level
    5. Take profit at 2-3x risk distance
    
    Args:
        data (pd.DataFrame): OHLCV data
        min_rr (float): Minimum risk/reward ratio required
        sr_lookback (int): Lookback for S/R detection
        sr_threshold (float): Threshold for S/R clustering
    
    Returns:
        pd.DataFrame: Data with signal columns
    """
    data = data.copy()
    
    # Detect swing points and S/R levels
    data = detect_swing_points(data, lookback=5)
    sr_data = detect_support_resistance(data, lookback=sr_lookback, threshold=sr_threshold)
    
    # Detect candlestick patterns
    data = detect_candlestick_patterns(data)
    
    # Initialize signal columns
    data['Signal'] = ''
    data['SignalType'] = ''
    data['Entry'] = np.nan
    data['StopLoss'] = np.nan
    data['TakeProfit'] = np.nan
    data['RR_Ratio'] = np.nan
    data['SignalStrength'] = ''
    
    support_levels = sr_data.get('support_levels', [])
    resistance_levels = sr_data.get('resistance_levels', [])
    
    # Combine and process zones
    all_zones = []
    for s in support_levels:
        all_zones.append({**s, 'type': 'support'})
    for r in resistance_levels:
        all_zones.append({**r, 'type': 'resistance'})
    
    # Check each bar for signals at levels
    for i in range(20, len(data)):
        row = data.iloc[i]
        
        # Skip if no pattern
        if row['Pattern'] not in ['PinBar', 'Engulfing', 'ImpulseCandle']:
            continue
        
        current_close = row['Close']
        current_high = row['High']
        current_low = row['Low']
        pattern = row['Pattern']
        pattern_dir = row['PatternDirection']
        
        # Check if price is at a level
        for zone in all_zones:
            level = zone['level']
            zone_range = zone['range']
            zone_low, zone_high = zone_range
            zone_strength = zone['strength']
            
            # Check proximity to level (within 2% or touching the zone)
            at_support = zone['type'] == 'support' and (
                current_low <= zone_high * 1.02 and current_low >= zone_low * 0.98
            )
            at_resistance = zone['type'] == 'resistance' and (
                current_high >= zone_low * 0.98 and current_high <= zone_high * 1.02
            )
            
            # Bullish signal at support
            if at_support and pattern_dir == 'Bullish':
                entry_price = current_close
                stop_price = zone_low * 0.99  # Stop below support
                risk = entry_price - stop_price
                target_price = entry_price + (risk * min_rr)
                
                rr = calculate_risk_reward(entry_price, stop_price, target_price)
                
                if rr >= min_rr:
                    data.iloc[i, data.columns.get_loc('Signal')] = 'BUY'
                    data.iloc[i, data.columns.get_loc('SignalType')] = f'Level+{pattern}'
                    data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                    data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                    data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                    data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                    data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Strong' if zone_strength >= 3 else 'Moderate'
            
            # Bearish signal at resistance
            elif at_resistance and pattern_dir == 'Bearish':
                entry_price = current_close
                stop_price = zone_high * 1.01  # Stop above resistance
                risk = stop_price - entry_price
                target_price = entry_price - (risk * min_rr)
                
                rr = calculate_risk_reward(entry_price, stop_price, target_price)
                
                if rr >= min_rr:
                    data.iloc[i, data.columns.get_loc('Signal')] = 'SELL'
                    data.iloc[i, data.columns.get_loc('SignalType')] = f'Level+{pattern}'
                    data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                    data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                    data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                    data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                    data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Strong' if zone_strength >= 3 else 'Moderate'
    
    return data, sr_data


def trend_pullback_strategy(data, ema_period=20, min_rr=2.0, swing_lookback=5):
    """
    Trend Pullback Strategy
    
    Enter trend continuation after pullbacks to key areas.
    
    Rules:
    1. Define trend using HH/HL (uptrend) or LH/LL (downtrend)
    2. Identify pullback zone (EMA or recent swing area)
    3. Enter when pullback prints signal candle in trend direction
    4. Stop beyond swing point
    5. Target next major structure level
    
    Args:
        data (pd.DataFrame): OHLCV data
        ema_period (int): EMA period for dynamic support/resistance
        min_rr (float): Minimum risk/reward ratio
        swing_lookback (int): Lookback for swing detection
    
    Returns:
        pd.DataFrame: Data with signal columns
    """
    data = data.copy()
    
    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Detect swing points and trend structure
    data = detect_swing_points(data, lookback=swing_lookback)
    data = identify_trend_structure(data, swing_lookback=swing_lookback)
    
    # Detect candlestick patterns
    data = detect_candlestick_patterns(data)
    
    # Initialize signal columns if not present
    if 'Signal' not in data.columns:
        data['Signal'] = ''
        data['SignalType'] = ''
        data['Entry'] = np.nan
        data['StopLoss'] = np.nan
        data['TakeProfit'] = np.nan
        data['RR_Ratio'] = np.nan
        data['SignalStrength'] = ''
    
    # Track recent swing lows/highs for stops and targets
    recent_swing_lows = []
    recent_swing_highs = []
    
    for i in range(ema_period + swing_lookback, len(data)):
        row = data.iloc[i]
        
        # Update swing point tracking
        if row.get('SwingLow', False):
            recent_swing_lows.append((i, row['Low']))
            if len(recent_swing_lows) > 3:
                recent_swing_lows.pop(0)
        
        if row.get('SwingHigh', False):
            recent_swing_highs.append((i, row['High']))
            if len(recent_swing_highs) > 3:
                recent_swing_highs.pop(0)
        
        trend = row.get('TrendStructure', 'Neutral')
        pattern = row.get('Pattern', '')
        pattern_dir = row.get('PatternDirection', '')
        
        current_close = row['Close']
        current_low = row['Low']
        current_high = row['High']
        ema_value = row['EMA']
        
        # Skip if no valid signal pattern
        if pattern not in ['PinBar', 'Engulfing', 'ImpulseCandle']:
            continue
        
        # Skip if signal already exists
        if row.get('Signal', '') != '':
            continue
        
        # Uptrend pullback setup
        if trend == 'Uptrend' and pattern_dir == 'Bullish':
            # Check if price pulled back to EMA zone
            ema_proximity = abs(current_low - ema_value) / ema_value < 0.02
            
            # Or pulled back to recent swing low area
            near_swing_low = False
            if recent_swing_lows:
                last_swing_low = recent_swing_lows[-1][1]
                near_swing_low = current_low <= last_swing_low * 1.03 and current_close > last_swing_low
            
            if ema_proximity or near_swing_low:
                entry_price = current_close
                
                # Stop below recent swing low
                if recent_swing_lows:
                    stop_price = recent_swing_lows[-1][1] * 0.99
                else:
                    stop_price = current_low * 0.99
                
                risk = entry_price - stop_price
                
                # Target at recent swing high or projected target
                if recent_swing_highs:
                    target_price = recent_swing_highs[-1][1]
                    if target_price <= entry_price:
                        target_price = entry_price + (risk * min_rr)
                else:
                    target_price = entry_price + (risk * min_rr)
                
                rr = calculate_risk_reward(entry_price, stop_price, target_price)
                
                if rr >= min_rr:
                    data.iloc[i, data.columns.get_loc('Signal')] = 'BUY'
                    data.iloc[i, data.columns.get_loc('SignalType')] = f'Pullback+{pattern}'
                    data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                    data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                    data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                    data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                    data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Strong' if trend == 'Uptrend' else 'Moderate'
        
        # Downtrend pullback setup
        elif trend == 'Downtrend' and pattern_dir == 'Bearish':
            # Check if price pulled back to EMA zone
            ema_proximity = abs(current_high - ema_value) / ema_value < 0.02
            
            # Or pulled back to recent swing high area
            near_swing_high = False
            if recent_swing_highs:
                last_swing_high = recent_swing_highs[-1][1]
                near_swing_high = current_high >= last_swing_high * 0.97 and current_close < last_swing_high
            
            if ema_proximity or near_swing_high:
                entry_price = current_close
                
                # Stop above recent swing high
                if recent_swing_highs:
                    stop_price = recent_swing_highs[-1][1] * 1.01
                else:
                    stop_price = current_high * 1.01
                
                risk = stop_price - entry_price
                
                # Target at recent swing low or projected target
                if recent_swing_lows:
                    target_price = recent_swing_lows[-1][1]
                    if target_price >= entry_price:
                        target_price = entry_price - (risk * min_rr)
                else:
                    target_price = entry_price - (risk * min_rr)
                
                rr = calculate_risk_reward(entry_price, stop_price, target_price)
                
                if rr >= min_rr:
                    data.iloc[i, data.columns.get_loc('Signal')] = 'SELL'
                    data.iloc[i, data.columns.get_loc('SignalType')] = f'Pullback+{pattern}'
                    data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                    data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                    data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                    data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                    data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Strong' if trend == 'Downtrend' else 'Moderate'
    
    return data


def breakout_retest_strategy(data, consolidation_bars=10, min_rr=2.0, confirmation_required=True):
    """
    Breakout & Retest Strategy
    
    Trade breakouts from consolidation zones with retest confirmation.
    
    Rules:
    1. Mark well-defined consolidation/range zones
    2. Wait for breakout with strong candle close
    3. Prefer entry on retest of broken level
    4. Confirm with signal candle
    5. Stop inside the zone, target measured move
    
    Args:
        data (pd.DataFrame): OHLCV data
        consolidation_bars (int): Minimum bars for consolidation zone
        min_rr (float): Minimum risk/reward ratio
        confirmation_required (bool): Require retest before entry
    
    Returns:
        pd.DataFrame: Data with signal columns
    """
    data = data.copy()
    
    # Detect market phase
    data = classify_market_phase(data, lookback=consolidation_bars)
    
    # Detect swing points for S/R
    data = detect_swing_points(data, lookback=5)
    sr_data = detect_support_resistance(data, lookback=consolidation_bars, threshold=0.02)
    
    # Detect breakouts and retests
    data = detect_breakouts_retests(data, sr_data, lookback=5)
    
    # Detect candlestick patterns
    data = detect_candlestick_patterns(data)
    
    # Initialize signal columns if not present
    if 'Signal' not in data.columns:
        data['Signal'] = ''
        data['SignalType'] = ''
        data['Entry'] = np.nan
        data['StopLoss'] = np.nan
        data['TakeProfit'] = np.nan
        data['RR_Ratio'] = np.nan
        data['SignalStrength'] = ''
    
    # Track recent breakouts for retest entries
    active_breakouts = []
    
    for i in range(consolidation_bars, len(data)):
        row = data.iloc[i]
        
        current_close = row['Close']
        current_high = row['High']
        current_low = row['Low']
        
        breakout_type = row.get('BreakoutType', '')
        breakout_level = row.get('BreakoutLevel', np.nan)
        is_retest = row.get('IsRetest', False)
        retest_level = row.get('RetestLevel', np.nan)
        pattern = row.get('Pattern', '')
        pattern_dir = row.get('PatternDirection', '')
        
        # Track new breakouts
        if breakout_type != '' and pd.notna(breakout_level):
            active_breakouts.append({
                'index': i,
                'type': breakout_type,
                'level': breakout_level,
                'range_size': row.get('Range', current_high - current_low)
            })
            
            # Keep only recent breakouts
            if len(active_breakouts) > 5:
                active_breakouts.pop(0)
            
            # Entry on breakout itself (if not requiring confirmation)
            if not confirmation_required:
                if breakout_type == 'Bullish':
                    entry_price = current_close
                    stop_price = breakout_level * 0.98
                    risk = entry_price - stop_price
                    target_price = entry_price + (risk * min_rr)
                    
                    rr = calculate_risk_reward(entry_price, stop_price, target_price)
                    
                    if rr >= min_rr and row.get('Signal', '') == '':
                        data.iloc[i, data.columns.get_loc('Signal')] = 'BUY'
                        data.iloc[i, data.columns.get_loc('SignalType')] = 'Breakout'
                        data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                        data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                        data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                        data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                        data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Moderate'
                
                elif breakout_type == 'Bearish':
                    entry_price = current_close
                    stop_price = breakout_level * 1.02
                    risk = stop_price - entry_price
                    target_price = entry_price - (risk * min_rr)
                    
                    rr = calculate_risk_reward(entry_price, stop_price, target_price)
                    
                    if rr >= min_rr and row.get('Signal', '') == '':
                        data.iloc[i, data.columns.get_loc('Signal')] = 'SELL'
                        data.iloc[i, data.columns.get_loc('SignalType')] = 'Breakout'
                        data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                        data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                        data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                        data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                        data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Moderate'
        
        # Entry on retest with confirmation
        if is_retest and pd.notna(retest_level):
            # Find the matching breakout
            matching_breakout = None
            for bo in active_breakouts:
                if abs(bo['level'] - retest_level) / retest_level < 0.02:
                    matching_breakout = bo
                    break
            
            if matching_breakout:
                # Check for confirming pattern
                has_confirmation = pattern in ['PinBar', 'Engulfing', 'ImpulseCandle']
                direction_match = (
                    (matching_breakout['type'] == 'Bullish' and pattern_dir == 'Bullish') or
                    (matching_breakout['type'] == 'Bearish' and pattern_dir == 'Bearish')
                )
                
                if has_confirmation and direction_match and row.get('Signal', '') == '':
                    if matching_breakout['type'] == 'Bullish':
                        entry_price = current_close
                        stop_price = retest_level * 0.98
                        risk = entry_price - stop_price
                        # Measured move target
                        target_price = entry_price + matching_breakout['range_size'] * min_rr
                        
                        rr = calculate_risk_reward(entry_price, stop_price, target_price)
                        
                        if rr >= min_rr:
                            data.iloc[i, data.columns.get_loc('Signal')] = 'BUY'
                            data.iloc[i, data.columns.get_loc('SignalType')] = f'Retest+{pattern}'
                            data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                            data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                            data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                            data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                            data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Strong'
                    
                    elif matching_breakout['type'] == 'Bearish':
                        entry_price = current_close
                        stop_price = retest_level * 1.02
                        risk = stop_price - entry_price
                        target_price = entry_price - matching_breakout['range_size'] * min_rr
                        
                        rr = calculate_risk_reward(entry_price, stop_price, target_price)
                        
                        if rr >= min_rr:
                            data.iloc[i, data.columns.get_loc('Signal')] = 'SELL'
                            data.iloc[i, data.columns.get_loc('SignalType')] = f'Retest+{pattern}'
                            data.iloc[i, data.columns.get_loc('Entry')] = entry_price
                            data.iloc[i, data.columns.get_loc('StopLoss')] = stop_price
                            data.iloc[i, data.columns.get_loc('TakeProfit')] = target_price
                            data.iloc[i, data.columns.get_loc('RR_Ratio')] = rr
                            data.iloc[i, data.columns.get_loc('SignalStrength')] = 'Strong'
    
    return data, sr_data


def generate_trading_signals(data, strategy_name, params=None):
    """
    Unified interface for generating trading signals.
    
    Args:
        data (pd.DataFrame): OHLCV data
        strategy_name (str): Name of strategy to use
        params (dict): Strategy parameters
    
    Returns:
        tuple: (data_with_signals, sr_data or None, signal_summary)
    """
    if params is None:
        params = {}
    
    sr_data = None
    
    if strategy_name == 'level_signal_rr':
        data, sr_data = level_signal_rr_strategy(
            data,
            min_rr=params.get('min_rr', 2.0),
            sr_lookback=params.get('sr_lookback', 20),
            sr_threshold=params.get('sr_threshold', 0.02)
        )
    
    elif strategy_name == 'trend_pullback':
        data = trend_pullback_strategy(
            data,
            ema_period=params.get('ema_period', 20),
            min_rr=params.get('min_rr', 2.0),
            swing_lookback=params.get('swing_lookback', 5)
        )
    
    elif strategy_name == 'breakout_retest':
        data, sr_data = breakout_retest_strategy(
            data,
            consolidation_bars=params.get('consolidation_bars', 10),
            min_rr=params.get('min_rr', 2.0),
            confirmation_required=params.get('confirmation_required', True)
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Generate signal summary
    signal_summary = get_signal_summary(data)
    
    return data, sr_data, signal_summary


def get_signal_summary(data):
    """
    Generate summary of trading signals.
    
    Args:
        data (pd.DataFrame): Data with signal columns
    
    Returns:
        dict: Signal summary statistics
    """
    if 'Signal' not in data.columns:
        return {}
    
    signals = data[data['Signal'] != '']
    
    return {
        'total_signals': len(signals),
        'buy_signals': len(signals[signals['Signal'] == 'BUY']),
        'sell_signals': len(signals[signals['Signal'] == 'SELL']),
        'avg_rr': signals['RR_Ratio'].mean() if len(signals) > 0 else 0,
        'strong_signals': len(signals[signals['SignalStrength'] == 'Strong']),
        'recent_signals': signals.tail(5)[['Signal', 'SignalType', 'Entry', 'StopLoss', 'TakeProfit', 'RR_Ratio']].to_dict('records')
    }


def combine_strategies(data, strategies, params=None):
    """
    Combine multiple strategies and aggregate signals.
    
    Args:
        data (pd.DataFrame): OHLCV data
        strategies (list): List of strategy names
        params (dict): Parameters for each strategy
    
    Returns:
        tuple: (combined_data, all_sr_data, combined_summary)
    """
    if params is None:
        params = {}
    
    combined_data = data.copy()
    all_sr_data = {}
    all_signals = []
    
    for strategy in strategies:
        strategy_params = params.get(strategy, {})
        result_data, sr_data, summary = generate_trading_signals(
            data.copy(), strategy, strategy_params
        )
        
        if sr_data:
            all_sr_data[strategy] = sr_data
        
        # Collect signals from this strategy
        strategy_signals = result_data[result_data['Signal'] != ''].copy()
        strategy_signals['Strategy'] = strategy
        all_signals.append(strategy_signals)
    
    # Combine all signals
    if all_signals:
        combined_signals = pd.concat(all_signals)
        combined_signals = combined_signals.sort_index()
        
        # Add signals back to combined data
        for idx in combined_signals.index:
            if combined_data.loc[idx, 'Signal'] if 'Signal' in combined_data.columns else '' == '':
                for col in ['Signal', 'SignalType', 'Entry', 'StopLoss', 'TakeProfit', 'RR_Ratio', 'SignalStrength']:
                    if col in combined_signals.columns:
                        combined_data.loc[idx, col] = combined_signals.loc[idx, col]
    
    combined_summary = get_signal_summary(combined_data)
    
    return combined_data, all_sr_data, combined_summary
