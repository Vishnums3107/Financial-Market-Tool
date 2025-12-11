"""
Paper Trading Module

Virtual portfolio for simulated trading:
- Paper trade execution
- Portfolio tracking
- P&L calculation
- Trade history
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


# Database path
PAPER_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'paper_trading.db')


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Order:
    """A paper trade order."""
    id: int = None
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    order_type: str = "market"  # market, limit
    limit_price: float = None
    fill_price: float = None
    status: str = "pending"
    created_at: str = ""
    filled_at: str = ""
    notes: str = ""


@dataclass
class TradeRecord:
    """Completed trade record."""
    id: int = None
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_date: str = ""
    exit_date: str = ""


class PaperTradingAccount:
    """
    Paper trading account for simulated trading.
    """
    
    def __init__(self, initial_balance: float = 100000.0, db_path: str = None):
        self.db_path = db_path or PAPER_DB_PATH
        self.initial_balance = initial_balance
        self._ensure_db_dir()
        self._init_db()
        self._ensure_account()
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _get_connection(self) -> sqlite3.Connection:
        # Add timeout to prevent locking issues
        return sqlite3.connect(self.db_path, timeout=10.0)
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                cash_balance REAL,
                initial_balance REAL,
                created_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                quantity INTEGER DEFAULT 0,
                avg_price REAL DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER,
                order_type TEXT DEFAULT 'market',
                limit_price REAL,
                fill_price REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                filled_at TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                entry_date TEXT,
                exit_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _ensure_account(self):
        """Ensure account record exists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM account WHERE id = 1")
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO account (id, cash_balance, initial_balance, created_at) VALUES (1, ?, ?, ?)",
                (self.initial_balance, self.initial_balance, datetime.now().isoformat())
            )
            conn.commit()
        
        conn.close()
    
    def get_cash_balance(self) -> float:
        """Get current cash balance."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cash_balance FROM account WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else self.initial_balance
    
    def _update_cash_balance(self, amount: float):
        """Update cash balance."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE account SET cash_balance = cash_balance + ? WHERE id = 1",
            (amount,)
        )
        conn.commit()
        conn.close()
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        order_type: str = "market",
        limit_price: float = None,
        notes: str = ""
    ) -> Optional[Order]:
        """
        Place a paper trade order.
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            current_price: Current market price
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            notes: Optional notes
        
        Returns:
            Order object if successful
        """
        side = side.upper()
        if side not in ['BUY', 'SELL']:
            raise ValueError("Side must be 'BUY' or 'SELL'")
        
        # For market orders, use current price
        fill_price = current_price if order_type == "market" else limit_price
        
        # Use a single connection for the entire operation
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Check cash balance for buy
            if side == 'BUY':
                cursor.execute("SELECT cash_balance FROM account WHERE id = 1")
                row = cursor.fetchone()
                cash = row[0] if row else 0
                cost = fill_price * quantity
                if cost > cash:
                    raise ValueError("Insufficient funds")
            
            # Check shares for sell
            if side == 'SELL':
                cursor.execute("SELECT quantity FROM positions WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                pos_qty = row[0] if row else 0
                if pos_qty < quantity:
                    raise ValueError("Insufficient shares")
            
            # Create order
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO orders (symbol, side, quantity, order_type, limit_price, 
                                  fill_price, status, created_at, filled_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, 'filled', ?, ?, ?)
            ''', (symbol, side, quantity, order_type, limit_price, fill_price, now, now, notes))
            
            order_id = cursor.lastrowid
            
            # Execute the trade within the same connection
            cursor.execute("SELECT quantity, avg_price FROM positions WHERE symbol = ?", (symbol,))
            pos_row = cursor.fetchone()
            
            if side == 'BUY':
                if pos_row:
                    old_qty, old_avg = pos_row
                    new_qty = old_qty + quantity
                    new_avg = ((old_qty * old_avg) + (quantity * fill_price)) / new_qty
                    cursor.execute(
                        "UPDATE positions SET quantity = ?, avg_price = ? WHERE symbol = ?",
                        (new_qty, new_avg, symbol)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO positions (symbol, quantity, avg_price) VALUES (?, ?, ?)",
                        (symbol, quantity, fill_price)
                    )
                
                # Deduct cash
                cursor.execute(
                    "UPDATE account SET cash_balance = cash_balance - ? WHERE id = 1",
                    (quantity * fill_price,)
                )
            
            else:  # SELL
                if pos_row:
                    old_qty, old_avg = pos_row
                    new_qty = old_qty - quantity
                    
                    pnl = (fill_price - old_avg) * quantity
                    pnl_pct = (fill_price - old_avg) / old_avg if old_avg > 0 else 0
                    
                    cursor.execute('''
                        INSERT INTO trades (symbol, side, quantity, entry_price, exit_price,
                                           pnl, pnl_pct, entry_date, exit_date)
                        VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, quantity, old_avg, fill_price, pnl, pnl_pct, now, now))
                    
                    if new_qty > 0:
                        cursor.execute(
                            "UPDATE positions SET quantity = ? WHERE symbol = ?",
                            (new_qty, symbol)
                        )
                    else:
                        cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                    
                    # Add cash
                    cursor.execute(
                        "UPDATE account SET cash_balance = cash_balance + ? WHERE id = 1",
                        (quantity * fill_price,)
                    )
            
            conn.commit()
            
            return Order(
                id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                fill_price=fill_price,
                status='filled',
                created_at=now,
                filled_at=now,
                notes=notes
            )
        
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _execute_order(self, symbol: str, side: str, quantity: int, price: float):
        """Execute the order - update position and cash."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Get current position
            cursor.execute("SELECT quantity, avg_price FROM positions WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            
            if side == 'BUY':
                if row:
                    # Update existing position
                    old_qty, old_avg = row
                    new_qty = old_qty + quantity
                    new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
                    cursor.execute(
                        "UPDATE positions SET quantity = ?, avg_price = ? WHERE symbol = ?",
                        (new_qty, new_avg, symbol)
                    )
                else:
                    # Create new position
                    cursor.execute(
                        "INSERT INTO positions (symbol, quantity, avg_price) VALUES (?, ?, ?)",
                        (symbol, quantity, price)
                    )
                
                # Deduct cash - do it in the same connection
                cursor.execute(
                    "UPDATE account SET cash_balance = cash_balance - ? WHERE id = 1",
                    (quantity * price,)
                )
            
            else:  # SELL
                if row:
                    old_qty, old_avg = row
                    new_qty = old_qty - quantity
                    
                    # Record realized P&L
                    pnl = (price - old_avg) * quantity
                    pnl_pct = (price - old_avg) / old_avg if old_avg > 0 else 0
                    
                    cursor.execute('''
                        INSERT INTO trades (symbol, side, quantity, entry_price, exit_price,
                                           pnl, pnl_pct, entry_date, exit_date)
                        VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, quantity, old_avg, price, pnl, pnl_pct,
                         datetime.now().isoformat(), datetime.now().isoformat()))
                    
                    if new_qty > 0:
                        cursor.execute(
                            "UPDATE positions SET quantity = ? WHERE symbol = ?",
                            (new_qty, symbol)
                        )
                    else:
                        cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                    
                    # Add cash from sale - do it in the same connection
                    cursor.execute(
                        "UPDATE account SET cash_balance = cash_balance + ? WHERE id = 1",
                        (quantity * price,)
                    )
            
            conn.commit()
        finally:
            conn.close()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT symbol, quantity, avg_price FROM positions WHERE symbol = ?",
            (symbol,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Position(
                symbol=row[0],
                quantity=row[1],
                avg_price=row[2]
            )
        return None
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbol, quantity, avg_price FROM positions WHERE quantity > 0")
        rows = cursor.fetchall()
        conn.close()
        
        return [Position(symbol=r[0], quantity=r[1], avg_price=r[2]) for r in rows]
    
    def update_position_prices(self, prices: Dict[str, float]):
        """Update current prices for positions."""
        positions = self.get_all_positions()
        for pos in positions:
            if pos.symbol in prices:
                pos.current_price = prices[pos.symbol]
                pos.unrealized_pnl = (pos.current_price - pos.avg_price) * pos.quantity
                pos.unrealized_pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
        return positions
    
    def get_portfolio_value(self, prices: Dict[str, float] = None) -> float:
        """Get total portfolio value (cash + positions)."""
        cash = self.get_cash_balance()
        
        positions = self.get_all_positions()
        positions_value = 0
        
        for pos in positions:
            price = prices.get(pos.symbol, pos.avg_price) if prices else pos.avg_price
            positions_value += pos.quantity * price
        
        return cash + positions_value
    
    def get_portfolio_summary(self, prices: Dict[str, float] = None) -> Dict:
        """Get portfolio summary."""
        cash = self.get_cash_balance()
        total_value = self.get_portfolio_value(prices)
        positions = self.get_all_positions()
        
        total_return = (total_value - self.initial_balance) / self.initial_balance
        
        return {
            'cash_balance': cash,
            'positions_value': total_value - cash,
            'total_value': total_value,
            'initial_balance': self.initial_balance,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_positions': len(positions)
        }
    
    def get_trade_history(self, limit: int = 100) -> List[TradeRecord]:
        """Get trade history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM trades ORDER BY exit_date DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        records = []
        for row in rows:
            columns = [d[0] for d in description]
            data = dict(zip(columns, row))
            records.append(TradeRecord(**data))
        
        return records
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        orders = []
        for row in rows:
            columns = [d[0] for d in description]
            data = dict(zip(columns, row))
            orders.append(Order(**data))
        
        return orders
    
    def reset_account(self):
        """Reset the paper trading account."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM positions")
        cursor.execute("DELETE FROM orders")
        cursor.execute("DELETE FROM trades")
        cursor.execute(
            "UPDATE account SET cash_balance = ? WHERE id = 1",
            (self.initial_balance,)
        )
        
        conn.commit()
        conn.close()


def format_positions_table(positions: List[Position]) -> pd.DataFrame:
    """Format positions for display."""
    if not positions:
        return pd.DataFrame(columns=['Symbol', 'Qty', 'Avg Price', 'Current', 'P&L', 'P&L %'])
    
    rows = []
    for p in positions:
        pnl_emoji = '🟢' if p.unrealized_pnl >= 0 else '🔴'
        rows.append({
            'Symbol': p.symbol,
            'Qty': p.quantity,
            'Avg Price': f"${p.avg_price:.2f}",
            'Current': f"${p.current_price:.2f}" if p.current_price else 'N/A',
            'P&L': f"{pnl_emoji} ${p.unrealized_pnl:,.2f}",
            'P&L %': f"{p.unrealized_pnl_pct*100:.1f}%"
        })
    
    return pd.DataFrame(rows)


def format_portfolio_summary(summary: Dict) -> pd.DataFrame:
    """Format portfolio summary for display."""
    return_emoji = '🟢' if summary['total_return'] >= 0 else '🔴'
    
    rows = [
        ('Cash Balance', f"${summary['cash_balance']:,.2f}"),
        ('Positions Value', f"${summary['positions_value']:,.2f}"),
        ('Total Value', f"${summary['total_value']:,.2f}"),
        ('Total Return', f"{return_emoji} {summary['total_return_pct']:.2f}%"),
        ('Open Positions', summary['num_positions'])
    ]
    
    return pd.DataFrame(rows, columns=['Metric', 'Value'])
