"""
Market Replay Module

Historical replay mode for practice and analysis:
- Bar-by-bar playback
- Decision tracking
- Performance analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ReplayState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class ReplayDecision:
    """A trading decision made during replay."""
    bar_index: int
    timestamp: str
    action: str  # BUY, SELL, HOLD
    price: float
    quantity: int = 0
    reason: str = ""


@dataclass
class ReplaySession:
    """A market replay session."""
    symbol: str
    start_date: str
    end_date: str
    decisions: List[ReplayDecision] = field(default_factory=list)
    current_bar: int = 0
    total_bars: int = 0
    pnl: float = 0.0
    position: int = 0
    avg_entry: float = 0.0


class MarketReplay:
    """
    Market replay engine for historical data playback.
    """
    
    def __init__(self, data: pd.DataFrame, symbol: str = "REPLAY"):
        self.data = data.copy()
        self.symbol = symbol
        self.total_bars = len(data)
        self.current_bar = 0
        self.state = ReplayState.STOPPED
        
        # Position tracking
        self.position = 0
        self.avg_entry = 0.0
        self.realized_pnl = 0.0
        self.decisions: List[ReplayDecision] = []
        
        # Initial capital
        self.initial_capital = 10000.0
        self.cash = self.initial_capital
    
    def reset(self):
        """Reset replay to beginning."""
        self.current_bar = 0
        self.state = ReplayState.STOPPED
        self.position = 0
        self.avg_entry = 0.0
        self.realized_pnl = 0.0
        self.decisions = []
        self.cash = self.initial_capital
    
    def get_visible_data(self) -> pd.DataFrame:
        """Get data visible up to current bar."""
        return self.data.iloc[:self.current_bar + 1].copy()
    
    def get_current_bar(self) -> pd.Series:
        """Get the current bar data."""
        if self.current_bar < self.total_bars:
            return self.data.iloc[self.current_bar]
        return None
    
    def get_current_price(self) -> float:
        """Get current close price."""
        bar = self.get_current_bar()
        return bar['Close'] if bar is not None else 0.0
    
    def step_forward(self) -> bool:
        """Advance to next bar."""
        if self.current_bar < self.total_bars - 1:
            self.current_bar += 1
            return True
        return False
    
    def step_backward(self) -> bool:
        """Go back one bar."""
        if self.current_bar > 0:
            self.current_bar -= 1
            return True
        return False
    
    def jump_to(self, bar_index: int) -> bool:
        """Jump to specific bar."""
        if 0 <= bar_index < self.total_bars:
            self.current_bar = bar_index
            return True
        return False
    
    def jump_to_date(self, target_date: str) -> bool:
        """Jump to specific date."""
        target = pd.to_datetime(target_date)
        for i, idx in enumerate(self.data.index):
            if idx >= target:
                self.current_bar = i
                return True
        return False
    
    def buy(self, quantity: int = 1, reason: str = "") -> bool:
        """Execute a buy order at current price."""
        price = self.get_current_price()
        if price <= 0:
            return False
        
        cost = price * quantity
        if cost > self.cash:
            return False
        
        # Update position
        if self.position >= 0:
            # Adding to long or opening new long
            total_cost = self.avg_entry * self.position + price * quantity
            self.position += quantity
            self.avg_entry = total_cost / self.position if self.position > 0 else 0
        else:
            # Covering short
            pnl = (self.avg_entry - price) * min(quantity, abs(self.position))
            self.realized_pnl += pnl
            self.cash += pnl
            self.position += quantity
            if self.position > 0:
                self.avg_entry = price
        
        self.cash -= cost
        
        # Record decision
        bar = self.get_current_bar()
        decision = ReplayDecision(
            bar_index=self.current_bar,
            timestamp=str(bar.name) if bar is not None else "",
            action='BUY',
            price=price,
            quantity=quantity,
            reason=reason
        )
        self.decisions.append(decision)
        
        return True
    
    def sell(self, quantity: int = 1, reason: str = "") -> bool:
        """Execute a sell order at current price."""
        price = self.get_current_price()
        if price <= 0:
            return False
        
        if quantity > self.position:
            quantity = self.position  # Can only sell what we have
        
        if quantity <= 0:
            return False
        
        # Calculate P&L
        pnl = (price - self.avg_entry) * quantity
        self.realized_pnl += pnl
        self.cash += price * quantity + pnl
        
        # Update position
        self.position -= quantity
        if self.position == 0:
            self.avg_entry = 0
        
        # Record decision
        bar = self.get_current_bar()
        decision = ReplayDecision(
            bar_index=self.current_bar,
            timestamp=str(bar.name) if bar is not None else "",
            action='SELL',
            price=price,
            quantity=quantity,
            reason=reason
        )
        self.decisions.append(decision)
        
        return True
    
    def close_position(self, reason: str = "Close all") -> bool:
        """Close entire position."""
        if self.position > 0:
            return self.sell(self.position, reason)
        elif self.position < 0:
            return self.buy(abs(self.position), reason)
        return True
    
    def get_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0:
            return 0.0
        current_price = self.get_current_price()
        return (current_price - self.avg_entry) * self.position
    
    def get_total_equity(self) -> float:
        """Get total equity (cash + unrealized)."""
        return self.cash + self.get_unrealized_pnl()
    
    def get_session_summary(self) -> Dict:
        """Get summary of replay session."""
        return {
            'symbol': self.symbol,
            'current_bar': self.current_bar,
            'total_bars': self.total_bars,
            'progress_pct': (self.current_bar / self.total_bars * 100) if self.total_bars > 0 else 0,
            'position': self.position,
            'avg_entry': self.avg_entry,
            'current_price': self.get_current_price(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.realized_pnl,
            'total_equity': self.get_total_equity(),
            'total_return_pct': ((self.get_total_equity() / self.initial_capital) - 1) * 100,
            'num_decisions': len(self.decisions),
            'num_buys': sum(1 for d in self.decisions if d.action == 'BUY'),
            'num_sells': sum(1 for d in self.decisions if d.action == 'SELL')
        }
    
    def get_decisions_df(self) -> pd.DataFrame:
        """Get decisions as DataFrame."""
        if not self.decisions:
            return pd.DataFrame(columns=['Time', 'Action', 'Price', 'Qty', 'Reason'])
        
        return pd.DataFrame([{
            'Time': d.timestamp[:10] if d.timestamp else '',
            'Action': d.action,
            'Price': f"${d.price:.2f}",
            'Qty': d.quantity,
            'Reason': d.reason
        } for d in self.decisions])
    
    def calculate_performance(self) -> Dict:
        """Calculate trading performance metrics."""
        if not self.decisions:
            return {}
        
        # Close any open position at current price
        final_equity = self.get_total_equity()
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Trade analysis
        buys = [d for d in self.decisions if d.action == 'BUY']
        sells = [d for d in self.decisions if d.action == 'SELL']
        
        return {
            'total_return_pct': total_return,
            'final_equity': final_equity,
            'num_trades': len(self.decisions),
            'num_buys': len(buys),
            'num_sells': len(sells),
            'bars_traded': self.current_bar,
            'decisions_per_bar': len(self.decisions) / self.current_bar if self.current_bar > 0 else 0
        }


def format_replay_summary(summary: Dict) -> pd.DataFrame:
    """Format replay summary for display."""
    pnl_emoji = '🟢' if summary.get('total_return_pct', 0) >= 0 else '🔴'
    
    rows = [
        ('Progress', f"{summary['progress_pct']:.0f}% ({summary['current_bar']}/{summary['total_bars']})"),
        ('Position', f"{summary['position']} shares"),
        ('Avg Entry', f"${summary['avg_entry']:.2f}" if summary['avg_entry'] > 0 else 'N/A'),
        ('Current Price', f"${summary['current_price']:.2f}"),
        ('Unrealized P&L', f"${summary['unrealized_pnl']:.2f}"),
        ('Realized P&L', f"${summary['realized_pnl']:.2f}"),
        ('Total Equity', f"${summary['total_equity']:.2f}"),
        ('Return', f"{pnl_emoji} {summary['total_return_pct']:.1f}%"),
    ]
    
    return pd.DataFrame(rows, columns=['Metric', 'Value'])
