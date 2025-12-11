"""
Risk Management Module

Tools for position sizing, portfolio risk analysis, and risk management:
- Position sizing calculators (fixed %, Kelly criterion)
- Correlation matrix analysis
- Portfolio heat map data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    shares: int
    position_value: float
    risk_amount: float
    risk_pct: float
    max_loss: float


def position_size_fixed_pct(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float
) -> PositionSizeResult:
    """
    Calculate position size using fixed percentage risk.
    
    Args:
        account_balance: Current account balance
        risk_pct: Percentage of account to risk (e.g., 0.02 for 2%)
        entry_price: Planned entry price
        stop_loss: Stop loss price
    
    Returns:
        PositionSizeResult with calculated position details
    """
    risk_amount = account_balance * risk_pct
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share <= 0:
        return PositionSizeResult(0, 0, 0, 0, 0)
    
    shares = int(risk_amount / risk_per_share)
    position_value = shares * entry_price
    max_loss = shares * risk_per_share
    
    return PositionSizeResult(
        shares=shares,
        position_value=position_value,
        risk_amount=risk_amount,
        risk_pct=risk_pct,
        max_loss=max_loss
    )


def position_size_kelly(
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    entry_price: float,
    kelly_fraction: float = 0.5
) -> PositionSizeResult:
    """
    Calculate position size using Kelly Criterion.
    
    Args:
        account_balance: Current account balance
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
        entry_price: Planned entry price
        kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
    
    Returns:
        PositionSizeResult with calculated position details
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return PositionSizeResult(0, 0, 0, 0, 0)
    
    # Kelly formula: f* = (bp - q) / b
    # where b = win/loss ratio, p = win rate, q = loss rate (1-p)
    win_loss_ratio = avg_win / avg_loss
    kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # Apply Kelly fraction (half Kelly is common for safety)
    adjusted_kelly = max(0, kelly_pct * kelly_fraction)
    
    # Calculate position
    position_value = account_balance * adjusted_kelly
    shares = int(position_value / entry_price)
    
    return PositionSizeResult(
        shares=shares,
        position_value=shares * entry_price,
        risk_amount=position_value,
        risk_pct=adjusted_kelly,
        max_loss=avg_loss * shares / entry_price if entry_price > 0 else 0
    )


def position_size_atr(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0
) -> PositionSizeResult:
    """
    Calculate position size using ATR-based stop.
    
    Args:
        account_balance: Current account balance
        risk_pct: Percentage of account to risk
        entry_price: Planned entry price
        atr: Current ATR value
        atr_multiplier: Multiple of ATR for stop distance
    
    Returns:
        PositionSizeResult with calculated position details
    """
    stop_distance = atr * atr_multiplier
    stop_loss = entry_price - stop_distance  # For long position
    
    return position_size_fixed_pct(account_balance, risk_pct, entry_price, stop_loss)


def calculate_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    period: int = 252
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        data_dict: Dict of {symbol: DataFrame with Close column}
        period: Number of periods for correlation calculation
    
    Returns:
        Correlation matrix as DataFrame
    """
    # Extract returns for each symbol
    returns = pd.DataFrame()
    
    for symbol, data in data_dict.items():
        if 'Close' in data.columns:
            close = data['Close'].tail(period)
            returns[symbol] = close.pct_change().dropna()
    
    if returns.empty or len(returns.columns) < 2:
        return pd.DataFrame()
    
    # Align all returns to common dates
    returns = returns.dropna()
    
    return returns.corr()


def calculate_rolling_correlation(
    data1: pd.Series,
    data2: pd.Series,
    window: int = 30
) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        data1: First price/return series
        data2: Second price/return series
        window: Rolling window size
    
    Returns:
        Series of rolling correlations
    """
    returns1 = data1.pct_change().dropna()
    returns2 = data2.pct_change().dropna()
    
    # Align series
    aligned = pd.concat([returns1, returns2], axis=1).dropna()
    aligned.columns = ['series1', 'series2']
    
    return aligned['series1'].rolling(window).corr(aligned['series2'])


def portfolio_var(
    positions: Dict[str, float],
    correlation_matrix: pd.DataFrame,
    volatilities: Dict[str, float],
    confidence: float = 0.95
) -> float:
    """
    Calculate portfolio Value at Risk (VaR).
    
    Args:
        positions: Dict of {symbol: position_value}
        correlation_matrix: Asset correlation matrix
        volatilities: Dict of {symbol: annualized_volatility}
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Portfolio VaR in dollar terms
    """
    from scipy import stats
    
    symbols = list(positions.keys())
    weights = np.array([positions[s] for s in symbols])
    total_value = np.sum(weights)
    
    if total_value <= 0:
        return 0
    
    weights = weights / total_value
    
    # Build covariance matrix
    vols = np.array([volatilities.get(s, 0.2) for s in symbols])
    corr = correlation_matrix.loc[symbols, symbols].values
    cov = np.outer(vols, vols) * corr
    
    # Portfolio volatility
    port_var = np.dot(weights.T, np.dot(cov, weights))
    port_std = np.sqrt(port_var)
    
    # Daily VaR (assuming 252 trading days)
    daily_std = port_std / np.sqrt(252)
    z_score = stats.norm.ppf(confidence)
    
    var = total_value * daily_std * z_score
    
    return var


def calculate_portfolio_heat_map_data(
    positions: Dict[str, Dict],
    sectors: Dict[str, str] = None
) -> Dict:
    """
    Calculate data for portfolio heat map visualization.
    
    Args:
        positions: Dict of {symbol: {'value': float, 'pnl': float, 'pnl_pct': float}}
        sectors: Optional dict mapping symbols to sectors
    
    Returns:
        Dict with data structured for heat map
    """
    if not positions:
        return {}
    
    total_value = sum(p.get('value', 0) for p in positions.values())
    
    # Calculate weights
    position_data = []
    for symbol, data in positions.items():
        value = data.get('value', 0)
        weight = value / total_value if total_value > 0 else 0
        
        position_data.append({
            'symbol': symbol,
            'value': value,
            'weight': weight,
            'pnl': data.get('pnl', 0),
            'pnl_pct': data.get('pnl_pct', 0),
            'sector': sectors.get(symbol, 'Unknown') if sectors else 'Unknown'
        })
    
    # Group by sector
    sector_data = {}
    for pos in position_data:
        sector = pos['sector']
        if sector not in sector_data:
            sector_data[sector] = {'value': 0, 'weight': 0, 'positions': []}
        sector_data[sector]['value'] += pos['value']
        sector_data[sector]['weight'] += pos['weight']
        sector_data[sector]['positions'].append(pos)
    
    return {
        'positions': position_data,
        'sectors': sector_data,
        'total_value': total_value
    }


def assess_portfolio_risk(
    positions: Dict[str, Dict],
    correlation_matrix: pd.DataFrame = None
) -> Dict:
    """
    Comprehensive portfolio risk assessment.
    
    Args:
        positions: Dict of position data
        correlation_matrix: Optional correlation matrix
    
    Returns:
        Dict with risk metrics
    """
    if not positions:
        return {}
    
    total_value = sum(p.get('value', 0) for p in positions.values())
    
    # Concentration risk
    weights = [p.get('value', 0) / total_value for p in positions.values() if total_value > 0]
    herfindahl = sum(w ** 2 for w in weights) if weights else 0
    effective_n = 1 / herfindahl if herfindahl > 0 else 0
    
    # Largest position
    max_weight = max(weights) if weights else 0
    
    # Correlation risk (if matrix provided)
    avg_correlation = 0
    if correlation_matrix is not None and not correlation_matrix.empty:
        symbols = [s for s in positions.keys() if s in correlation_matrix.index]
        if len(symbols) > 1:
            corr_subset = correlation_matrix.loc[symbols, symbols]
            # Get upper triangle excluding diagonal
            mask = np.triu(np.ones(corr_subset.shape), k=1).astype(bool)
            avg_correlation = corr_subset.values[mask].mean()
    
    risk_score = 0
    
    # Score concentration (0-33 points)
    if max_weight > 0.3:
        risk_score += 33
    elif max_weight > 0.2:
        risk_score += 22
    elif max_weight > 0.1:
        risk_score += 11
    
    # Score diversification (0-33 points)
    if effective_n < 3:
        risk_score += 33
    elif effective_n < 5:
        risk_score += 22
    elif effective_n < 10:
        risk_score += 11
    
    # Score correlation (0-34 points)
    if avg_correlation > 0.7:
        risk_score += 34
    elif avg_correlation > 0.5:
        risk_score += 22
    elif avg_correlation > 0.3:
        risk_score += 11
    
    risk_level = 'Low' if risk_score < 33 else 'Medium' if risk_score < 66 else 'High'
    
    return {
        'total_value': total_value,
        'n_positions': len(positions),
        'effective_positions': effective_n,
        'herfindahl_index': herfindahl,
        'max_weight': max_weight,
        'avg_correlation': avg_correlation,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'recommendations': _generate_risk_recommendations(max_weight, effective_n, avg_correlation)
    }


def _generate_risk_recommendations(max_weight: float, effective_n: float, avg_corr: float) -> List[str]:
    """Generate risk management recommendations."""
    recommendations = []
    
    if max_weight > 0.2:
        recommendations.append(f"Consider reducing largest position (currently {max_weight*100:.1f}% of portfolio)")
    
    if effective_n < 5:
        recommendations.append(f"Low diversification - consider adding more uncorrelated positions")
    
    if avg_corr > 0.5:
        recommendations.append("High average correlation - portfolio is exposed to common risk factors")
    
    if not recommendations:
        recommendations.append("Portfolio risk metrics are within acceptable ranges")
    
    return recommendations


def format_position_size_table(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_levels: List[float] = [0.01, 0.02, 0.03, 0.05]
) -> pd.DataFrame:
    """Generate position sizing table for different risk levels."""
    rows = []
    
    for risk_pct in risk_levels:
        result = position_size_fixed_pct(account_balance, risk_pct, entry_price, stop_loss)
        rows.append({
            'Risk %': f"{risk_pct * 100:.1f}%",
            'Shares': result.shares,
            'Position Value': f"${result.position_value:,.2f}",
            'Max Loss': f"${result.max_loss:,.2f}"
        })
    
    return pd.DataFrame(rows)
