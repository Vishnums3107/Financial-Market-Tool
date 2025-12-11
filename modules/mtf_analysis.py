"""
Multi-Timeframe Analysis Module

Tools for multi-timeframe analysis:
- Timeframe aggregation from lower to higher timeframes
- MTF trend confirmation
- Confluence scoring system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


# Timeframe definitions in minutes
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1H': 60,
    '4H': 240,
    '1D': 1440,
    '1W': 10080,
    '1M': 43200  # Approximate
}


def aggregate_timeframe(
    data: pd.DataFrame,
    target_tf: str = '1H',
    source_tf: str = '1m'
) -> pd.DataFrame:
    """
    Aggregate data from a lower timeframe to a higher timeframe.
    
    Args:
        data: DataFrame with OHLCV data and DatetimeIndex
        target_tf: Target timeframe ('5m', '15m', '30m', '1H', '4H', '1D', '1W', '1M')
        source_tf: Source timeframe of input data
    
    Returns:
        Aggregated DataFrame
    """
    if target_tf not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe: {target_tf}")
    
    # Map to pandas resample string
    resample_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1H': '1h',
        '4H': '4h',
        '1D': '1D',
        '1W': '1W',
        '1M': '1ME'
    }
    
    resample_str = resample_map.get(target_tf, '1D')
    
    # Aggregate OHLCV
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
    
    resampled = data.resample(resample_str).agg(agg_dict)
    
    # Drop rows with NaN values (incomplete periods)
    resampled = resampled.dropna()
    
    return resampled


def get_multiple_timeframes(
    data: pd.DataFrame,
    timeframes: List[str] = ['1H', '4H', '1D']
) -> Dict[str, pd.DataFrame]:
    """
    Get data for multiple timeframes from source data.
    
    Args:
        data: Source DataFrame (preferably 1m or 5m data)
        timeframes: List of target timeframes
    
    Returns:
        Dict of {timeframe: DataFrame}
    """
    result = {}
    
    for tf in timeframes:
        try:
            result[tf] = aggregate_timeframe(data, tf)
        except Exception as e:
            print(f"Error aggregating to {tf}: {e}")
    
    return result


def calculate_ema_slope(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate EMA slope direction.
    
    Returns:
        Series with 1 (bullish), -1 (bearish), 0 (neutral)
    """
    ema = data['Close'].ewm(span=period, adjust=False).mean()
    slope = ema.diff()
    
    result = pd.Series(0, index=data.index)
    result[slope > 0] = 1
    result[slope < 0] = -1
    
    return result


def identify_mtf_trend(
    data_dict: Dict[str, pd.DataFrame],
    ema_period: int = 20
) -> Dict:
    """
    Identify trend direction across multiple timeframes.
    
    Args:
        data_dict: Dict of {timeframe: DataFrame}
        ema_period: EMA period for trend identification
    
    Returns:
        Dict with trend info for each timeframe
    """
    results = {}
    
    for tf, data in data_dict.items():
        if len(data) < ema_period:
            results[tf] = {
                'trend': 'Unknown',
                'strength': 0,
                'ema_slope': 0
            }
            continue
        
        # Calculate EMA
        ema = data['Close'].ewm(span=ema_period, adjust=False).mean()
        current_price = data['Close'].iloc[-1]
        current_ema = ema.iloc[-1]
        
        # EMA slope over last few bars
        slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5] if len(ema) >= 5 else 0
        
        # Determine trend
        if current_price > current_ema and slope > 0:
            trend = 'Uptrend'
            strength = min(100, int(abs(slope) * 1000))
        elif current_price < current_ema and slope < 0:
            trend = 'Downtrend'
            strength = min(100, int(abs(slope) * 1000))
        else:
            trend = 'Ranging'
            strength = 0
        
        results[tf] = {
            'trend': trend,
            'strength': strength,
            'ema_slope': slope,
            'price_vs_ema': (current_price - current_ema) / current_ema * 100
        }
    
    return results


def calculate_confluence_score(
    mtf_trends: Dict,
    timeframe_weights: Dict[str, float] = None
) -> Dict:
    """
    Calculate confluence score based on trend alignment across timeframes.
    
    Args:
        mtf_trends: Dict from identify_mtf_trend()
        timeframe_weights: Optional weights for each timeframe
    
    Returns:
        Dict with confluence score and details
    """
    if not mtf_trends:
        return {'score': 0, 'direction': 'Neutral', 'aligned': False}
    
    # Default weights (higher timeframes have more weight)
    default_weights = {
        '1m': 0.05, '5m': 0.1, '15m': 0.15, '30m': 0.2,
        '1H': 0.25, '4H': 0.4, '1D': 0.6, '1W': 0.8, '1M': 1.0
    }
    weights = timeframe_weights or default_weights
    
    bullish_score = 0
    bearish_score = 0
    total_weight = 0
    
    alignment = []
    
    for tf, trend_data in mtf_trends.items():
        weight = weights.get(tf, 0.5)
        total_weight += weight
        
        trend = trend_data.get('trend', 'Ranging')
        strength = trend_data.get('strength', 0) / 100
        
        if trend == 'Uptrend':
            bullish_score += weight * (1 + strength)
            alignment.append((tf, 'Bullish'))
        elif trend == 'Downtrend':
            bearish_score += weight * (1 + strength)
            alignment.append((tf, 'Bearish'))
        else:
            alignment.append((tf, 'Neutral'))
    
    # Normalize scores
    if total_weight > 0:
        bullish_score /= total_weight
        bearish_score /= total_weight
    
    # Determine overall direction and score
    if bullish_score > bearish_score:
        direction = 'Bullish'
        score = int(bullish_score * 100)
    elif bearish_score > bullish_score:
        direction = 'Bearish'
        score = int(bearish_score * 100)
    else:
        direction = 'Neutral'
        score = 0
    
    # Check alignment
    bullish_count = sum(1 for _, d in alignment if d == 'Bullish')
    bearish_count = sum(1 for _, d in alignment if d == 'Bearish')
    aligned = bullish_count == len(alignment) or bearish_count == len(alignment)
    
    return {
        'score': min(100, score),
        'direction': direction,
        'bullish_score': int(bullish_score * 100),
        'bearish_score': int(bearish_score * 100),
        'aligned': aligned,
        'alignment_details': alignment
    }


def get_mtf_entry_signal(
    higher_tf_data: pd.DataFrame,
    lower_tf_data: pd.DataFrame,
    higher_ema_period: int = 20,
    signal_lookback: int = 5
) -> Dict:
    """
    Get entry signal based on higher timeframe trend and lower timeframe trigger.
    
    Args:
        higher_tf_data: Higher timeframe DataFrame
        lower_tf_data: Lower timeframe DataFrame
        higher_ema_period: EMA period for higher TF trend
        signal_lookback: Bars to look back for trigger
    
    Returns:
        Dict with signal information
    """
    # Determine higher TF trend
    higher_ema = higher_tf_data['Close'].ewm(span=higher_ema_period, adjust=False).mean()
    higher_price = higher_tf_data['Close'].iloc[-1]
    higher_trend = 'Bullish' if higher_price > higher_ema.iloc[-1] else 'Bearish'
    
    # Look for lower TF trigger
    lower_ema = lower_tf_data['Close'].ewm(span=higher_ema_period, adjust=False).mean()
    
    signal = None
    
    # Check for pullback to EMA
    recent_data = lower_tf_data.tail(signal_lookback)
    
    if higher_trend == 'Bullish':
        # Look for bullish trigger: price pulled back to EMA and bouncing
        touched_ema = any(recent_data['Low'] <= lower_ema.iloc[-signal_lookback:])
        bouncing = lower_tf_data['Close'].iloc[-1] > lower_ema.iloc[-1]
        
        if touched_ema and bouncing:
            signal = {
                'type': 'BUY',
                'reason': 'Bullish higher TF + pullback to EMA on lower TF',
                'confidence': 'High' if bouncing else 'Medium'
            }
    
    elif higher_trend == 'Bearish':
        # Look for bearish trigger: price rallied to EMA and rejecting
        touched_ema = any(recent_data['High'] >= lower_ema.iloc[-signal_lookback:])
        rejecting = lower_tf_data['Close'].iloc[-1] < lower_ema.iloc[-1]
        
        if touched_ema and rejecting:
            signal = {
                'type': 'SELL',
                'reason': 'Bearish higher TF + rally to EMA on lower TF',
                'confidence': 'High' if rejecting else 'Medium'
            }
    
    return {
        'higher_tf_trend': higher_trend,
        'signal': signal,
        'higher_ema': higher_ema.iloc[-1],
        'lower_ema': lower_ema.iloc[-1]
    }


def analyze_mtf(
    data: pd.DataFrame,
    timeframes: List[str] = ['1H', '4H', '1D'],
    ema_period: int = 20
) -> Dict:
    """
    Complete multi-timeframe analysis.
    
    Args:
        data: Source DataFrame (should be fine-grained, e.g., 1m or 5m)
        timeframes: Timeframes to analyze
        ema_period: EMA period for trend detection
    
    Returns:
        Dict with complete MTF analysis
    """
    # Get data for all timeframes
    mtf_data = get_multiple_timeframes(data, timeframes)
    
    # Add source data as the finest timeframe
    mtf_data['Source'] = data
    
    # Identify trends
    trends = identify_mtf_trend(mtf_data, ema_period)
    
    # Calculate confluence
    confluence = calculate_confluence_score(trends)
    
    # Get entry signal if we have at least 2 timeframes
    entry_signal = None
    if len(timeframes) >= 2:
        higher_tf = timeframes[-1]  # Highest timeframe
        lower_tf = timeframes[0]   # Lowest of selected
        
        if higher_tf in mtf_data and lower_tf in mtf_data:
            entry_signal = get_mtf_entry_signal(
                mtf_data[higher_tf],
                mtf_data[lower_tf],
                ema_period
            )
    
    return {
        'timeframes_analyzed': list(mtf_data.keys()),
        'trends': trends,
        'confluence': confluence,
        'entry_signal': entry_signal,
        'recommendation': _generate_mtf_recommendation(confluence, entry_signal)
    }


def _generate_mtf_recommendation(confluence: Dict, entry_signal: Dict) -> str:
    """Generate trading recommendation based on MTF analysis."""
    score = confluence.get('score', 0)
    direction = confluence.get('direction', 'Neutral')
    aligned = confluence.get('aligned', False)
    
    if score < 30 or direction == 'Neutral':
        return "⚠️ No clear direction - wait for better setup"
    
    if aligned and score >= 70:
        if entry_signal and entry_signal.get('signal'):
            signal = entry_signal['signal']
            return f"✅ Strong {direction} bias with {signal['type']} trigger - High conviction setup"
        return f"🔎 Strong {direction} bias - wait for lower TF entry trigger"
    
    if score >= 50:
        return f"📊 Moderate {direction} bias ({score}/100) - proceed with caution"
    
    return "⏸️ Mixed signals - no trade recommended"


def format_mtf_summary(analysis: Dict) -> pd.DataFrame:
    """Format MTF analysis as a display table."""
    trends = analysis.get('trends', {})
    
    rows = []
    for tf, data in trends.items():
        trend = data.get('trend', 'Unknown')
        strength = data.get('strength', 0)
        
        emoji = '🟢' if trend == 'Uptrend' else '🔴' if trend == 'Downtrend' else '🟡'
        
        rows.append({
            'Timeframe': tf,
            'Trend': f"{emoji} {trend}",
            'Strength': f"{strength}%"
        })
    
    confluence = analysis.get('confluence', {})
    rows.append({
        'Timeframe': '📊 Confluence',
        'Trend': confluence.get('direction', 'Neutral'),
        'Strength': f"{confluence.get('score', 0)}/100"
    })
    
    return pd.DataFrame(rows)
