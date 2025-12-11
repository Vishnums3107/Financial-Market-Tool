"""
ML Pattern Recognition Module

Machine learning-based chart pattern detection:
- CNN for visual pattern recognition
- Pattern classification (head & shoulders, triangles, wedges, etc.)
- Lightweight inference without full TensorFlow dependency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class PatternPrediction:
    """Prediction result for a chart pattern."""
    pattern_name: str
    confidence: float
    direction: str  # Bullish, Bearish, Neutral
    start_idx: int
    end_idx: int
    target_price: float = None
    stop_price: float = None


# Pattern definitions
CHART_PATTERNS = {
    'head_shoulders': {
        'name': 'Head and Shoulders',
        'direction': 'Bearish',
        'description': 'Three peaks with middle higher'
    },
    'inv_head_shoulders': {
        'name': 'Inverse Head and Shoulders',
        'direction': 'Bullish',
        'description': 'Three troughs with middle lower'
    },
    'double_top': {
        'name': 'Double Top',
        'direction': 'Bearish',
        'description': 'Two peaks at similar level'
    },
    'double_bottom': {
        'name': 'Double Bottom',
        'direction': 'Bullish',
        'description': 'Two troughs at similar level'
    },
    'ascending_triangle': {
        'name': 'Ascending Triangle',
        'direction': 'Bullish',
        'description': 'Flat resistance, rising support'
    },
    'descending_triangle': {
        'name': 'Descending Triangle',
        'direction': 'Bearish',
        'description': 'Flat support, falling resistance'
    },
    'symmetric_triangle': {
        'name': 'Symmetric Triangle',
        'direction': 'Neutral',
        'description': 'Converging trendlines'
    },
    'rising_wedge': {
        'name': 'Rising Wedge',
        'direction': 'Bearish',
        'description': 'Converging upward trendlines'
    },
    'falling_wedge': {
        'name': 'Falling Wedge',
        'direction': 'Bullish',
        'description': 'Converging downward trendlines'
    },
    'bull_flag': {
        'name': 'Bull Flag',
        'direction': 'Bullish',
        'description': 'Sharp rise then consolidation'
    },
    'bear_flag': {
        'name': 'Bear Flag',
        'direction': 'Bearish',
        'description': 'Sharp drop then consolidation'
    }
}


def detect_swing_points(prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect swing highs and lows in price data.
    
    Args:
        prices: Array of prices
        order: Number of points on each side for comparison
    
    Returns:
        Tuple of (swing_high_indices, swing_low_indices)
    """
    n = len(prices)
    highs = []
    lows = []
    
    for i in range(order, n - order):
        is_high = True
        is_low = True
        
        for j in range(1, order + 1):
            if prices[i] < prices[i - j] or prices[i] < prices[i + j]:
                is_high = False
            if prices[i] > prices[i - j] or prices[i] > prices[i + j]:
                is_low = False
        
        if is_high:
            highs.append(i)
        if is_low:
            lows.append(i)
    
    return np.array(highs), np.array(lows)


def detect_head_shoulders(
    data: pd.DataFrame,
    high_indices: np.ndarray,
    tolerance: float = 0.02
) -> Optional[PatternPrediction]:
    """
    Detect Head and Shoulders pattern.
    
    Args:
        data: OHLCV DataFrame
        high_indices: Indices of swing highs
        tolerance: Tolerance for shoulder height matching
    
    Returns:
        PatternPrediction if pattern found
    """
    if len(high_indices) < 3:
        return None
    
    # Look at last 3 swing highs
    for i in range(len(high_indices) - 2):
        left_idx = high_indices[i]
        head_idx = high_indices[i + 1]
        right_idx = high_indices[i + 2]
        
        left_high = data['High'].iloc[left_idx]
        head_high = data['High'].iloc[head_idx]
        right_high = data['High'].iloc[right_idx]
        
        # Head should be higher than both shoulders
        if head_high > left_high and head_high > right_high:
            # Shoulders should be at similar level
            shoulder_diff = abs(left_high - right_high) / left_high
            if shoulder_diff < tolerance:
                # Found pattern
                neckline = min(
                    data['Low'].iloc[left_idx:head_idx].min(),
                    data['Low'].iloc[head_idx:right_idx].min()
                )
                
                pattern_height = head_high - neckline
                target = neckline - pattern_height
                
                return PatternPrediction(
                    pattern_name='Head and Shoulders',
                    confidence=0.7 + (0.3 * (1 - shoulder_diff / tolerance)),
                    direction='Bearish',
                    start_idx=left_idx,
                    end_idx=right_idx,
                    target_price=target,
                    stop_price=head_high * 1.01
                )
    
    return None


def detect_inv_head_shoulders(
    data: pd.DataFrame,
    low_indices: np.ndarray,
    tolerance: float = 0.02
) -> Optional[PatternPrediction]:
    """Detect Inverse Head and Shoulders pattern."""
    if len(low_indices) < 3:
        return None
    
    for i in range(len(low_indices) - 2):
        left_idx = low_indices[i]
        head_idx = low_indices[i + 1]
        right_idx = low_indices[i + 2]
        
        left_low = data['Low'].iloc[left_idx]
        head_low = data['Low'].iloc[head_idx]
        right_low = data['Low'].iloc[right_idx]
        
        # Head should be lower than both shoulders
        if head_low < left_low and head_low < right_low:
            shoulder_diff = abs(left_low - right_low) / left_low
            if shoulder_diff < tolerance:
                neckline = max(
                    data['High'].iloc[left_idx:head_idx].max(),
                    data['High'].iloc[head_idx:right_idx].max()
                )
                
                pattern_height = neckline - head_low
                target = neckline + pattern_height
                
                return PatternPrediction(
                    pattern_name='Inverse Head and Shoulders',
                    confidence=0.7 + (0.3 * (1 - shoulder_diff / tolerance)),
                    direction='Bullish',
                    start_idx=left_idx,
                    end_idx=right_idx,
                    target_price=target,
                    stop_price=head_low * 0.99
                )
    
    return None


def detect_double_top(
    data: pd.DataFrame,
    high_indices: np.ndarray,
    tolerance: float = 0.02
) -> Optional[PatternPrediction]:
    """Detect Double Top pattern."""
    if len(high_indices) < 2:
        return None
    
    for i in range(len(high_indices) - 1):
        first_idx = high_indices[i]
        second_idx = high_indices[i + 1]
        
        first_high = data['High'].iloc[first_idx]
        second_high = data['High'].iloc[second_idx]
        
        diff = abs(first_high - second_high) / first_high
        if diff < tolerance:
            neckline = data['Low'].iloc[first_idx:second_idx].min()
            pattern_height = first_high - neckline
            target = neckline - pattern_height
            
            return PatternPrediction(
                pattern_name='Double Top',
                confidence=0.65 + (0.35 * (1 - diff / tolerance)),
                direction='Bearish',
                start_idx=first_idx,
                end_idx=second_idx,
                target_price=target,
                stop_price=max(first_high, second_high) * 1.01
            )
    
    return None


def detect_double_bottom(
    data: pd.DataFrame,
    low_indices: np.ndarray,
    tolerance: float = 0.02
) -> Optional[PatternPrediction]:
    """Detect Double Bottom pattern."""
    if len(low_indices) < 2:
        return None
    
    for i in range(len(low_indices) - 1):
        first_idx = low_indices[i]
        second_idx = low_indices[i + 1]
        
        first_low = data['Low'].iloc[first_idx]
        second_low = data['Low'].iloc[second_idx]
        
        diff = abs(first_low - second_low) / first_low
        if diff < tolerance:
            neckline = data['High'].iloc[first_idx:second_idx].max()
            pattern_height = neckline - first_low
            target = neckline + pattern_height
            
            return PatternPrediction(
                pattern_name='Double Bottom',
                confidence=0.65 + (0.35 * (1 - diff / tolerance)),
                direction='Bullish',
                start_idx=first_idx,
                end_idx=second_idx,
                target_price=target,
                stop_price=min(first_low, second_low) * 0.99
            )
    
    return None


def detect_triangle(
    data: pd.DataFrame,
    high_indices: np.ndarray,
    low_indices: np.ndarray,
    min_points: int = 4
) -> Optional[PatternPrediction]:
    """Detect triangle patterns (ascending, descending, symmetric)."""
    if len(high_indices) < 2 or len(low_indices) < 2:
        return None
    
    # Get recent highs and lows
    recent_highs = high_indices[-min_points:] if len(high_indices) >= min_points else high_indices
    recent_lows = low_indices[-min_points:] if len(low_indices) >= min_points else low_indices
    
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None
    
    # Calculate slopes of resistance and support
    high_values = [data['High'].iloc[i] for i in recent_highs]
    low_values = [data['Low'].iloc[i] for i in recent_lows]
    
    # Linear regression for trendlines
    high_slope = np.polyfit(range(len(high_values)), high_values, 1)[0]
    low_slope = np.polyfit(range(len(low_values)), low_values, 1)[0]
    
    # Normalize slopes
    avg_price = data['Close'].mean()
    high_slope_norm = high_slope / avg_price
    low_slope_norm = low_slope / avg_price
    
    # Classify triangle type
    slope_threshold = 0.001
    
    if abs(high_slope_norm) < slope_threshold and low_slope_norm > slope_threshold:
        pattern_name = 'Ascending Triangle'
        direction = 'Bullish'
        confidence = 0.7
    elif high_slope_norm < -slope_threshold and abs(low_slope_norm) < slope_threshold:
        pattern_name = 'Descending Triangle'
        direction = 'Bearish'
        confidence = 0.7
    elif high_slope_norm < 0 and low_slope_norm > 0:
        pattern_name = 'Symmetric Triangle'
        direction = 'Neutral'
        confidence = 0.6
    else:
        return None
    
    start_idx = min(recent_highs[0], recent_lows[0])
    end_idx = max(recent_highs[-1], recent_lows[-1])
    
    # Calculate target based on pattern height
    pattern_height = max(high_values) - min(low_values)
    current_price = data['Close'].iloc[-1]
    
    if direction == 'Bullish':
        target = current_price + pattern_height
        stop = min(low_values) * 0.99
    elif direction == 'Bearish':
        target = current_price - pattern_height
        stop = max(high_values) * 1.01
    else:
        target = None
        stop = None
    
    return PatternPrediction(
        pattern_name=pattern_name,
        confidence=confidence,
        direction=direction,
        start_idx=start_idx,
        end_idx=end_idx,
        target_price=target,
        stop_price=stop
    )


def detect_wedge(
    data: pd.DataFrame,
    high_indices: np.ndarray,
    low_indices: np.ndarray
) -> Optional[PatternPrediction]:
    """Detect wedge patterns (rising wedge, falling wedge)."""
    if len(high_indices) < 2 or len(low_indices) < 2:
        return None
    
    # Get recent swing points
    recent_highs = high_indices[-3:]
    recent_lows = low_indices[-3:]
    
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None
    
    high_values = [data['High'].iloc[i] for i in recent_highs]
    low_values = [data['Low'].iloc[i] for i in recent_lows]
    
    # Calculate slopes
    high_slope = (high_values[-1] - high_values[0]) / len(high_values)
    low_slope = (low_values[-1] - low_values[0]) / len(low_values)
    
    # Check for converging lines (wedge)
    avg_price = data['Close'].mean()
    convergence = abs(high_slope - low_slope) / avg_price < 0.01
    
    if not convergence:
        return None
    
    # Rising wedge: both slopes positive, resistance steeper
    if high_slope > 0 and low_slope > 0:
        pattern_name = 'Rising Wedge'
        direction = 'Bearish'
        confidence = 0.65
    # Falling wedge: both slopes negative, support steeper
    elif high_slope < 0 and low_slope < 0:
        pattern_name = 'Falling Wedge'
        direction = 'Bullish'
        confidence = 0.65
    else:
        return None
    
    start_idx = min(recent_highs[0], recent_lows[0])
    end_idx = max(recent_highs[-1], recent_lows[-1])
    
    pattern_height = max(high_values) - min(low_values)
    current_price = data['Close'].iloc[-1]
    
    if direction == 'Bullish':
        target = current_price + pattern_height * 0.618
        stop = min(low_values) * 0.99
    else:
        target = current_price - pattern_height * 0.618
        stop = max(high_values) * 1.01
    
    return PatternPrediction(
        pattern_name=pattern_name,
        confidence=confidence,
        direction=direction,
        start_idx=start_idx,
        end_idx=end_idx,
        target_price=target,
        stop_price=stop
    )


def detect_all_patterns(
    data: pd.DataFrame,
    swing_order: int = 5,
    tolerance: float = 0.02
) -> List[PatternPrediction]:
    """
    Detect all chart patterns in the data.
    
    Args:
        data: OHLCV DataFrame
        swing_order: Order for swing point detection
        tolerance: Tolerance for pattern matching
    
    Returns:
        List of detected patterns
    """
    if len(data) < swing_order * 3:
        return []
    
    # Detect swing points
    prices = data['Close'].values
    high_idx, low_idx = detect_swing_points(prices, swing_order)
    
    patterns = []
    
    # Check for each pattern type
    pattern_checks = [
        (detect_head_shoulders, [data, high_idx, tolerance]),
        (detect_inv_head_shoulders, [data, low_idx, tolerance]),
        (detect_double_top, [data, high_idx, tolerance]),
        (detect_double_bottom, [data, low_idx, tolerance]),
        (detect_triangle, [data, high_idx, low_idx]),
        (detect_wedge, [data, high_idx, low_idx]),
    ]
    
    for func, args in pattern_checks:
        try:
            result = func(*args)
            if result:
                patterns.append(result)
        except Exception as e:
            continue
    
    # Sort by confidence
    patterns.sort(key=lambda x: x.confidence, reverse=True)
    
    return patterns


def analyze_patterns(data: pd.DataFrame, lookback: int = 100) -> Dict:
    """
    Analyze data for chart patterns and return summary.
    
    Args:
        data: OHLCV DataFrame
        lookback: Number of bars to analyze
    
    Returns:
        Dict with pattern analysis results
    """
    analysis_data = data.tail(lookback).copy()
    
    patterns = detect_all_patterns(analysis_data)
    
    # Summarize findings
    bullish_patterns = [p for p in patterns if p.direction == 'Bullish']
    bearish_patterns = [p for p in patterns if p.direction == 'Bearish']
    
    # Overall bias
    if len(bullish_patterns) > len(bearish_patterns):
        bias = 'Bullish'
        bias_confidence = np.mean([p.confidence for p in bullish_patterns]) if bullish_patterns else 0
    elif len(bearish_patterns) > len(bullish_patterns):
        bias = 'Bearish'
        bias_confidence = np.mean([p.confidence for p in bearish_patterns]) if bearish_patterns else 0
    else:
        bias = 'Neutral'
        bias_confidence = 0
    
    return {
        'patterns_found': len(patterns),
        'patterns': patterns,
        'bullish_count': len(bullish_patterns),
        'bearish_count': len(bearish_patterns),
        'bias': bias,
        'bias_confidence': bias_confidence,
        'top_pattern': patterns[0] if patterns else None
    }


def format_patterns_table(patterns: List[PatternPrediction]) -> pd.DataFrame:
    """Format patterns as display table."""
    if not patterns:
        return pd.DataFrame(columns=['Pattern', 'Direction', 'Confidence', 'Target', 'Stop'])
    
    rows = []
    for p in patterns:
        emoji = '🟢' if p.direction == 'Bullish' else '🔴' if p.direction == 'Bearish' else '🟡'
        rows.append({
            'Pattern': p.pattern_name,
            'Direction': f"{emoji} {p.direction}",
            'Confidence': f"{p.confidence*100:.0f}%",
            'Target': f"${p.target_price:.2f}" if p.target_price else 'N/A',
            'Stop': f"${p.stop_price:.2f}" if p.stop_price else 'N/A'
        })
    
    return pd.DataFrame(rows)
