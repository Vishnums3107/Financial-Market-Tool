"""
Monte Carlo Simulation Module

Risk analysis through trade sequence simulation:
- Equity curve confidence intervals
- Risk of ruin calculation
- Distribution analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    simulations: np.ndarray  # Shape: (n_simulations, n_trades)
    final_equities: np.ndarray
    percentiles: Dict[int, float]
    risk_of_ruin: float
    expected_final: float
    worst_case: float
    best_case: float


def monte_carlo_simulation(
    trade_results: List[float],
    initial_capital: float = 10000.0,
    n_simulations: int = 1000,
    n_trades: int = None,
    ruin_threshold: float = 0.5
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation by shuffling and resampling trades.
    
    Args:
        trade_results: List of P&L values from historical trades
        initial_capital: Starting capital
        n_simulations: Number of simulation runs
        n_trades: Number of trades per simulation (default: same as input)
        ruin_threshold: Equity level considered "ruin" (e.g., 0.5 = 50% of initial)
    
    Returns:
        MonteCarloResult with simulation data and statistics
    """
    if not trade_results or len(trade_results) < 2:
        return MonteCarloResult(
            simulations=np.array([]),
            final_equities=np.array([]),
            percentiles={},
            risk_of_ruin=0,
            expected_final=initial_capital,
            worst_case=initial_capital,
            best_case=initial_capital
        )
    
    trades = np.array(trade_results)
    n_trades = n_trades or len(trades)
    ruin_level = initial_capital * ruin_threshold
    
    # Run simulations
    simulations = np.zeros((n_simulations, n_trades))
    final_equities = np.zeros(n_simulations)
    ruin_count = 0
    
    for i in range(n_simulations):
        # Resample trades with replacement
        sampled_trades = np.random.choice(trades, size=n_trades, replace=True)
        
        # Calculate equity curve
        equity = initial_capital
        min_equity = initial_capital
        
        for j, pnl in enumerate(sampled_trades):
            equity += pnl
            simulations[i, j] = equity
            min_equity = min(min_equity, equity)
            
            # Check for ruin
            if equity <= ruin_level:
                ruin_count += 1
                break
        
        final_equities[i] = equity
    
    # Calculate statistics
    percentiles = {
        5: np.percentile(final_equities, 5),
        25: np.percentile(final_equities, 25),
        50: np.percentile(final_equities, 50),
        75: np.percentile(final_equities, 75),
        95: np.percentile(final_equities, 95)
    }
    
    risk_of_ruin = ruin_count / n_simulations
    
    return MonteCarloResult(
        simulations=simulations,
        final_equities=final_equities,
        percentiles=percentiles,
        risk_of_ruin=risk_of_ruin,
        expected_final=np.mean(final_equities),
        worst_case=np.min(final_equities),
        best_case=np.max(final_equities)
    )


def calculate_confidence_bands(
    simulations: np.ndarray,
    confidence_levels: List[int] = [5, 25, 50, 75, 95]
) -> Dict[int, np.ndarray]:
    """
    Calculate confidence bands from simulation results.
    
    Args:
        simulations: Array of shape (n_simulations, n_trades)
        confidence_levels: Percentile levels to calculate
    
    Returns:
        Dict mapping percentile to equity curve array
    """
    bands = {}
    for level in confidence_levels:
        bands[level] = np.percentile(simulations, level, axis=0)
    return bands


def analyze_drawdown_distribution(
    simulations: np.ndarray,
    initial_capital: float = 10000.0
) -> Dict:
    """
    Analyze drawdown distribution across simulations.
    
    Args:
        simulations: Array of shape (n_simulations, n_trades)
        initial_capital: Starting capital
    
    Returns:
        Dict with drawdown statistics
    """
    n_simulations = simulations.shape[0]
    max_drawdowns = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        equity_curve = simulations[i]
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax
        max_drawdowns[i] = np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    return {
        'avg_max_drawdown': np.mean(max_drawdowns),
        'median_max_drawdown': np.median(max_drawdowns),
        'worst_drawdown': np.min(max_drawdowns),
        'best_drawdown': np.max(max_drawdowns),
        'drawdown_std': np.std(max_drawdowns),
        'percentile_5': np.percentile(max_drawdowns, 5),
        'percentile_95': np.percentile(max_drawdowns, 95)
    }


def calculate_expected_value(
    trade_results: List[float],
    n_future_trades: int = 100
) -> Dict:
    """
    Calculate expected value statistics for future trades.
    
    Args:
        trade_results: Historical trade P&L values
        n_future_trades: Number of future trades to project
    
    Returns:
        Dict with expected value statistics
    """
    if not trade_results:
        return {}
    
    trades = np.array(trade_results)
    
    # Basic statistics
    mean_trade = np.mean(trades)
    std_trade = np.std(trades)
    
    # Win/loss breakdown
    winners = trades[trades > 0]
    losers = trades[trades < 0]
    
    win_rate = len(winners) / len(trades) if len(trades) > 0 else 0
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(losers) if len(losers) > 0 else 0
    
    # Expected value per trade
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
    
    # Kelly criterion
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
    else:
        kelly_pct = 0
    
    return {
        'mean_trade': mean_trade,
        'std_trade': std_trade,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expected_value': expected_value,
        'kelly_criterion': max(0, kelly_pct),  # Don't recommend negative sizing
        'projected_pnl': expected_value * n_future_trades,
        'projected_std': std_trade * np.sqrt(n_future_trades)
    }


def run_sensitivity_analysis(
    trade_results: List[float],
    initial_capital: float = 10000.0,
    position_sizes: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
    n_simulations: int = 500
) -> pd.DataFrame:
    """
    Analyze sensitivity of results to position sizing.
    
    Args:
        trade_results: List of P&L values
        initial_capital: Starting capital
        position_sizes: Multipliers to apply to trades
        n_simulations: Simulations per size
    
    Returns:
        DataFrame with results for each position size
    """
    results = []
    
    for size in position_sizes:
        scaled_trades = [t * size for t in trade_results]
        mc_result = monte_carlo_simulation(
            scaled_trades, 
            initial_capital, 
            n_simulations
        )
        
        results.append({
            'position_size': size,
            'expected_final': mc_result.expected_final,
            'percentile_5': mc_result.percentiles[5],
            'percentile_50': mc_result.percentiles[50],
            'percentile_95': mc_result.percentiles[95],
            'risk_of_ruin': mc_result.risk_of_ruin,
            'worst_case': mc_result.worst_case,
            'best_case': mc_result.best_case
        })
    
    return pd.DataFrame(results)


def format_monte_carlo_summary(result: MonteCarloResult, initial_capital: float) -> pd.DataFrame:
    """Format Monte Carlo results as a display table."""
    summary_data = [
        ('Expected Final Equity', f"${result.expected_final:,.2f}"),
        ('5th Percentile (Worst Case)', f"${result.percentiles[5]:,.2f}"),
        ('25th Percentile', f"${result.percentiles[25]:,.2f}"),
        ('50th Percentile (Median)', f"${result.percentiles[50]:,.2f}"),
        ('75th Percentile', f"${result.percentiles[75]:,.2f}"),
        ('95th Percentile (Best Case)', f"${result.percentiles[95]:,.2f}"),
        ('Risk of Ruin', f"{result.risk_of_ruin * 100:.1f}%"),
        ('Expected Return', f"{((result.expected_final / initial_capital) - 1) * 100:.1f}%"),
    ]
    
    return pd.DataFrame(summary_data, columns=['Metric', 'Value'])
