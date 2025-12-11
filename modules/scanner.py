"""
Stock Scanner Module

Watchlist scanning for trading opportunities:
- Multi-stock setup scanning
- Pattern detection across symbols
- Signal detection
- Opportunity ranking
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings


@dataclass
class ScanResult:
    """Result of scanning a single stock."""
    symbol: str
    score: float
    signals: List[str]
    patterns: List[str]
    trend: str
    current_price: float
    change_pct: float
    volume_ratio: float
    at_support: bool
    at_resistance: bool
    timestamp: str


@dataclass
class ScanCriteria:
    """Criteria for stock scanning."""
    min_price: float = 1.0
    max_price: float = 10000.0
    min_volume: int = 100000
    trend_filter: str = None  # 'Uptrend', 'Downtrend', None for all
    pattern_filter: List[str] = None
    at_sr_level: bool = False
    has_signal: bool = False
    min_score: float = 0.0


def scan_single_stock(
    symbol: str,
    data: pd.DataFrame,
    criteria: ScanCriteria = None
) -> Optional[ScanResult]:
    """
    Scan a single stock for opportunities.
    
    Args:
        symbol: Stock symbol
        data: OHLCV data for the stock
        criteria: Optional filter criteria
    
    Returns:
        ScanResult if stock matches criteria
    """
    from modules.price_action import analyze_price_action
    from modules.strategies import get_signal_summary
    
    criteria = criteria or ScanCriteria()
    
    if len(data) < 20:
        return None
    
    try:
        # Basic price and volume checks
        current_price = data['Close'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if current_price < criteria.min_price or current_price > criteria.max_price:
            return None
        if avg_volume < criteria.min_volume:
            return None
        
        # Run price action analysis
        analyzed_data, sr_data, pa_summary = analyze_price_action(data)
        
        # Get trend
        trend = pa_summary.get('trend', 'Unknown')
        if criteria.trend_filter and trend != criteria.trend_filter:
            return None
        
        # Check patterns
        patterns = []
        if 'Pattern' in analyzed_data.columns:
            recent_patterns = analyzed_data[analyzed_data['Pattern'] != ''].tail(5)
            patterns = recent_patterns['Pattern'].tolist()
        
        if criteria.pattern_filter:
            if not any(p in patterns for p in criteria.pattern_filter):
                return None
        
        # Check S/R proximity
        support_levels = sr_data.get('support_levels', [])
        resistance_levels = sr_data.get('resistance_levels', [])
        
        at_support = False
        at_resistance = False
        
        for s in support_levels[:3]:
            if abs(current_price - s['level']) / s['level'] < 0.02:
                at_support = True
                break
        
        for r in resistance_levels[:3]:
            if abs(current_price - r['level']) / r['level'] < 0.02:
                at_resistance = True
                break
        
        if criteria.at_sr_level and not (at_support or at_resistance):
            return None
        
        # Check for signals
        signals = []
        if 'Signal' in analyzed_data.columns:
            signal_data = analyzed_data[analyzed_data['Signal'] != ''].tail(3)
            signals = signal_data['Signal'].tolist()
        
        if criteria.has_signal and not signals:
            return None
        
        # Calculate score
        score = calculate_opportunity_score(
            trend=trend,
            patterns=patterns,
            signals=signals,
            at_support=at_support,
            at_resistance=at_resistance,
            volume_ratio=volume_ratio
        )
        
        if score < criteria.min_score:
            return None
        
        # Calculate price change
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100
        
        return ScanResult(
            symbol=symbol,
            score=score,
            signals=signals,
            patterns=patterns,
            trend=trend,
            current_price=current_price,
            change_pct=change_pct,
            volume_ratio=volume_ratio,
            at_support=at_support,
            at_resistance=at_resistance,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        warnings.warn(f"Error scanning {symbol}: {e}")
        return None


def calculate_opportunity_score(
    trend: str,
    patterns: List[str],
    signals: List[str],
    at_support: bool,
    at_resistance: bool,
    volume_ratio: float
) -> float:
    """
    Calculate opportunity score (0-100).
    
    Args:
        trend: Current trend
        patterns: Detected patterns
        signals: Trading signals
        at_support: Price at support
        at_resistance: Price at resistance
        volume_ratio: Volume vs average
    
    Returns:
        Score from 0-100
    """
    score = 0
    
    # Trend score (0-25)
    if trend in ['Uptrend', 'Downtrend']:
        score += 25
    elif trend == 'Ranging':
        score += 10
    
    # Pattern score (0-25)
    pattern_weights = {
        'Engulfing': 15,
        'PinBar': 12,
        'InsideBar': 8,
        'ImpulseCandle': 10,
        'Doji': 5
    }
    for pattern in patterns:
        score += pattern_weights.get(pattern, 5)
    score = min(score, 50)  # Cap pattern contribution
    
    # Signal score (0-25)
    if signals:
        score += 25
    
    # S/R proximity (0-15)
    if at_support or at_resistance:
        score += 15
    
    # Volume score (0-10)
    if volume_ratio > 2.0:
        score += 10
    elif volume_ratio > 1.5:
        score += 7
    elif volume_ratio > 1.2:
        score += 4
    
    return min(100, score)


def scan_watchlist(
    symbols: List[str],
    fetch_func: Callable,
    criteria: ScanCriteria = None,
    max_workers: int = 5
) -> List[ScanResult]:
    """
    Scan multiple stocks for opportunities.
    
    Args:
        symbols: List of stock symbols
        fetch_func: Function to fetch data for a symbol
        criteria: Scan criteria
        max_workers: Max concurrent scans
    
    Returns:
        List of ScanResults, sorted by score
    """
    results = []
    
    def scan_symbol(symbol):
        try:
            data = fetch_func(symbol)
            if data is not None and not data.empty:
                return scan_single_stock(symbol, data, criteria)
        except Exception as e:
            warnings.warn(f"Failed to scan {symbol}: {e}")
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results


def quick_scan_universe(
    universe: str = 'sp500',
    criteria: ScanCriteria = None,
    max_stocks: int = 50
) -> List[str]:
    """
    Get a list of stocks to scan based on universe.
    
    Args:
        universe: 'sp500', 'nasdaq100', 'dow30', 'custom'
        criteria: Optional criteria
        max_stocks: Maximum stocks to return
    
    Returns:
        List of stock symbols
    """
    universes = {
        'dow30': [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
            'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
            'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 
            'VZ', 'WBA', 'WMT'
        ],
        'tech': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL'
        ],
        'finance': [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA',
            'BLK', 'SCHW', 'USB'
        ],
        'healthcare': [
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT',
            'DHR', 'BMY', 'AMGN', 'GILD'
        ],
        'energy': [
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
            'OXY', 'HAL'
        ]
    }
    
    symbols = universes.get(universe, universes['dow30'])
    return symbols[:max_stocks]


def format_scan_results(results: List[ScanResult]) -> pd.DataFrame:
    """Format scan results for display."""
    if not results:
        return pd.DataFrame(columns=['Symbol', 'Score', 'Trend', 'Price', 'Change', 'Signals'])
    
    rows = []
    for r in results:
        trend_emoji = '🟢' if r.trend == 'Uptrend' else '🔴' if r.trend == 'Downtrend' else '🟡'
        change_emoji = '📈' if r.change_pct > 0 else '📉'
        sr_label = '🎯S' if r.at_support else ('🎯R' if r.at_resistance else '')
        
        rows.append({
            'Symbol': r.symbol,
            'Score': f"{r.score:.0f}",
            'Trend': f"{trend_emoji} {r.trend}",
            'Price': f"${r.current_price:.2f}",
            'Change': f"{change_emoji} {r.change_pct:+.1f}%",
            'Vol Ratio': f"{r.volume_ratio:.1f}x",
            'S/R': sr_label,
            'Signals': ', '.join(r.signals) if r.signals else '-'
        })
    
    return pd.DataFrame(rows)


def get_top_opportunities(
    results: List[ScanResult],
    top_n: int = 10,
    direction: str = None
) -> List[ScanResult]:
    """
    Get top opportunities from scan results.
    
    Args:
        results: Scan results
        top_n: Number of top results
        direction: 'Bullish' or 'Bearish' filter
    
    Returns:
        Top N results
    """
    filtered = results
    
    if direction == 'Bullish':
        filtered = [r for r in results if r.trend in ['Uptrend'] or r.at_support]
    elif direction == 'Bearish':
        filtered = [r for r in results if r.trend in ['Downtrend'] or r.at_resistance]
    
    return filtered[:top_n]
