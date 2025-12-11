"""
Backtesting Engine Module

Complete backtesting framework for strategy validation with:
- Historical trade simulation with slippage and commissions
- Walk-forward analysis for robust optimization
- Comprehensive performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: OrderSide
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    r_multiple: float
    commission: float
    slippage: float
    stop_loss: float = None
    take_profit: float = None
    signal_type: str = ""
    

@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: OrderSide
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float = None
    take_profit: float = None
    signal_type: str = ""
    status: PositionStatus = PositionStatus.OPEN


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission_per_share: float = 0.01
    commission_min: float = 1.0
    slippage_pct: float = 0.001  # 0.1%
    max_position_pct: float = 0.1  # 10% of capital per position
    use_stop_loss: bool = True
    use_take_profit: bool = True
    risk_per_trade_pct: float = 0.02  # 2% risk per trade


class BacktestEngine:
    """
    Core backtesting engine for strategy validation.
    
    Features:
    - Realistic trade simulation with slippage and commissions
    - Position management with stop loss and take profit
    - Comprehensive performance metrics
    - Walk-forward analysis support
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        """Reset the backtest state."""
        self.capital = self.config.initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_date = None
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """
        Run a complete backtest on the provided data and signals.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            signals: DataFrame with Signal, Entry, StopLoss, TakeProfit columns
        
        Returns:
            Dict with trades, equity curve, and metrics
        """
        self.reset()
        
        # Merge signals with data
        if 'Signal' not in data.columns:
            data = data.join(signals[['Signal', 'Entry', 'StopLoss', 'TakeProfit', 'SignalType']], 
                           how='left')
            data['Signal'] = data['Signal'].fillna('')
        
        # Iterate through each bar
        for idx, row in data.iterrows():
            self.current_date = idx
            current_price = row['Close']
            
            # Check for stop loss / take profit on open positions
            self._check_exits(row)
            
            # Process new signals
            if row.get('Signal') in ['BUY', 'SELL']:
                self._process_signal(row)
            
            # Record equity
            self._record_equity(current_price)
        
        # Close any remaining positions at last price
        self._close_all_positions(data['Close'].iloc[-1])
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        return {
            'trades': self.trades,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'metrics': metrics,
            'final_capital': self.capital
        }
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to entry/exit price."""
        slippage = price * self.config.slippage_pct
        if side == OrderSide.BUY:
            return price + slippage  # Worse fill for buy
        else:
            return price - slippage  # Worse fill for sell
    
    def _calculate_commission(self, quantity: int) -> float:
        """Calculate commission for a trade."""
        commission = quantity * self.config.commission_per_share
        return max(commission, self.config.commission_min)
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management."""
        risk_amount = self.capital * self.config.risk_per_trade_pct
        
        if stop_loss and stop_loss > 0:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                quantity = int(risk_amount / risk_per_share)
            else:
                quantity = int((self.capital * self.config.max_position_pct) / entry_price)
        else:
            # No stop loss, use max position percentage
            quantity = int((self.capital * self.config.max_position_pct) / entry_price)
        
        # Ensure we can afford it
        max_affordable = int(self.capital / entry_price)
        quantity = min(quantity, max_affordable)
        
        return max(1, quantity)  # At least 1 share
    
    def _process_signal(self, row: pd.Series):
        """Process a trading signal."""
        signal = row['Signal']
        entry_price = row.get('Entry', row['Close'])
        stop_loss = row.get('StopLoss', None)
        take_profit = row.get('TakeProfit', None)
        signal_type = row.get('SignalType', '')
        
        # Check if we already have a position
        if self.positions:
            # Close existing position if opposite signal
            existing = self.positions[0]
            if (signal == 'BUY' and existing.side == OrderSide.SELL) or \
               (signal == 'SELL' and existing.side == OrderSide.BUY):
                self._close_position(existing, row['Close'], "Signal Reversal")
        
        # Open new position
        side = OrderSide.BUY if signal == 'BUY' else OrderSide.SELL
        fill_price = self._apply_slippage(entry_price, side)
        quantity = self._calculate_position_size(fill_price, stop_loss)
        
        # Check if we have enough capital
        cost = fill_price * quantity + self._calculate_commission(quantity)
        if cost > self.capital:
            return  # Can't afford this trade
        
        # Deduct commission
        commission = self._calculate_commission(quantity)
        self.capital -= commission
        
        # Create position
        position = Position(
            symbol="BACKTEST",
            side=side,
            entry_date=self.current_date,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type=signal_type
        )
        self.positions.append(position)
    
    def _check_exits(self, row: pd.Series):
        """Check if any positions should be exited."""
        positions_to_close = []
        
        for position in self.positions:
            high = row['High']
            low = row['Low']
            
            if position.side == OrderSide.BUY:
                # Check stop loss
                if position.stop_loss and low <= position.stop_loss:
                    positions_to_close.append((position, position.stop_loss, "Stop Loss"))
                # Check take profit
                elif position.take_profit and high >= position.take_profit:
                    positions_to_close.append((position, position.take_profit, "Take Profit"))
            else:  # SELL
                # Check stop loss
                if position.stop_loss and high >= position.stop_loss:
                    positions_to_close.append((position, position.stop_loss, "Stop Loss"))
                # Check take profit
                elif position.take_profit and low <= position.take_profit:
                    positions_to_close.append((position, position.take_profit, "Take Profit"))
        
        for position, exit_price, reason in positions_to_close:
            self._close_position(position, exit_price, reason)
    
    def _close_position(self, position: Position, exit_price: float, reason: str = ""):
        """Close a position and record the trade."""
        fill_price = self._apply_slippage(exit_price, 
            OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY)
        
        commission = self._calculate_commission(position.quantity)
        slippage = abs(exit_price - fill_price) * position.quantity
        
        # Calculate P&L
        if position.side == OrderSide.BUY:
            pnl = (fill_price - position.entry_price) * position.quantity - commission
        else:
            pnl = (position.entry_price - fill_price) * position.quantity - commission
        
        pnl_pct = pnl / (position.entry_price * position.quantity)
        
        # Calculate R-multiple
        if position.stop_loss:
            risk = abs(position.entry_price - position.stop_loss)
            if position.side == OrderSide.BUY:
                reward = fill_price - position.entry_price
            else:
                reward = position.entry_price - fill_price
            r_multiple = reward / risk if risk > 0 else 0
        else:
            r_multiple = 0
        
        # Record trade
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=self.current_date,
            exit_price=fill_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            commission=commission,
            slippage=slippage,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            signal_type=position.signal_type
        )
        self.trades.append(trade)
        
        # Update capital
        self.capital += pnl + (position.entry_price * position.quantity)
        
        # Remove position
        self.positions.remove(position)
    
    def _close_all_positions(self, price: float):
        """Close all open positions."""
        for position in list(self.positions):
            self._close_position(position, price, "End of Backtest")
    
    def _record_equity(self, current_price: float):
        """Record current equity for equity curve."""
        # Calculate unrealized P&L
        unrealized = 0
        for position in self.positions:
            if position.side == OrderSide.BUY:
                unrealized += (current_price - position.entry_price) * position.quantity
            else:
                unrealized += (position.entry_price - current_price) * position.quantity
        
        equity = self.capital + unrealized
        self.equity_curve.append({
            'date': self.current_date,
            'equity': equity,
            'capital': self.capital,
            'unrealized': unrealized,
            'open_positions': len(self.positions)
        })
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {'total_trades': 0, 'message': 'No trades executed'}
        
        trades_df = pd.DataFrame([{
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'r_multiple': t.r_multiple,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'side': t.side.value
        } for t in self.trades])
        
        # Basic metrics
        total_trades = len(self.trades)
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_winner = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loser = losers['pnl'].mean() if len(losers) > 0 else 0
        
        # Profit factor
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate equity curve metrics
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Annualized return (assuming 252 trading days)
            days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
            years = days / 365.25 if days > 0 else 1
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility (annualized)
            daily_vol = equity_df['returns'].std()
            annualized_vol = daily_vol * np.sqrt(252) if pd.notna(daily_vol) else 0
            
            # Sharpe Ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = equity_df['returns'][equity_df['returns'] < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
            
            # Maximum Drawdown
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min()
            
            # Calmar Ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            annualized_return = 0
            annualized_vol = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
        
        # R-multiple stats
        avg_r = trades_df['r_multiple'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_r_multiple': avg_r,
            'initial_capital': self.config.initial_capital,
            'final_capital': self.capital
        }


def walk_forward_analysis(
    data: pd.DataFrame,
    strategy_func: Callable,
    train_pct: float = 0.7,
    n_splits: int = 5,
    config: BacktestConfig = None
) -> Dict:
    """
    Perform walk-forward analysis for robust strategy validation.
    
    Args:
        data: Full historical data
        strategy_func: Function that generates signals from data
        train_pct: Percentage of each window for training
        n_splits: Number of walk-forward periods
        config: Backtest configuration
    
    Returns:
        Dict with in-sample and out-of-sample results
    """
    results = []
    n_samples = len(data)
    window_size = n_samples // n_splits
    
    for i in range(n_splits):
        start_idx = i * window_size
        end_idx = min((i + 2) * window_size, n_samples)
        
        window_data = data.iloc[start_idx:end_idx].copy()
        train_size = int(len(window_data) * train_pct)
        
        train_data = window_data.iloc[:train_size]
        test_data = window_data.iloc[train_size:]
        
        if len(train_data) < 30 or len(test_data) < 10:
            continue
        
        # Generate signals on training data (in-sample)
        train_signals = strategy_func(train_data.copy())
        
        # Apply strategy to test data (out-of-sample)
        test_signals = strategy_func(test_data.copy())
        
        # Backtest both
        engine = BacktestEngine(config)
        
        train_result = engine.run_backtest(train_data, train_signals)
        test_result = engine.run_backtest(test_data, test_signals)
        
        results.append({
            'period': i + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'in_sample': train_result['metrics'],
            'out_of_sample': test_result['metrics']
        })
    
    # Aggregate results
    if results:
        oos_metrics = [r['out_of_sample'] for r in results]
        avg_oos_return = np.mean([m.get('total_return', 0) for m in oos_metrics])
        avg_oos_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in oos_metrics])
        avg_oos_win_rate = np.mean([m.get('win_rate', 0) for m in oos_metrics])
        
        return {
            'periods': results,
            'summary': {
                'avg_oos_return': avg_oos_return,
                'avg_oos_sharpe': avg_oos_sharpe,
                'avg_oos_win_rate': avg_oos_win_rate,
                'n_periods': len(results)
            }
        }
    
    return {'periods': [], 'summary': {}}


def format_metrics_table(metrics: Dict) -> pd.DataFrame:
    """Format metrics dictionary as a display table."""
    display_metrics = [
        ('Total Trades', metrics.get('total_trades', 0), ''),
        ('Win Rate', metrics.get('win_rate', 0) * 100, '%'),
        ('Profit Factor', metrics.get('profit_factor', 0), 'x'),
        ('Total Return', metrics.get('total_return', 0) * 100, '%'),
        ('Annualized Return', metrics.get('annualized_return', 0) * 100, '%'),
        ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), ''),
        ('Sortino Ratio', metrics.get('sortino_ratio', 0), ''),
        ('Calmar Ratio', metrics.get('calmar_ratio', 0), ''),
        ('Max Drawdown', metrics.get('max_drawdown', 0) * 100, '%'),
        ('Avg R-Multiple', metrics.get('avg_r_multiple', 0), 'R'),
        ('Final Capital', metrics.get('final_capital', 0), '$'),
    ]
    
    return pd.DataFrame(display_metrics, columns=['Metric', 'Value', 'Unit'])
