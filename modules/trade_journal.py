"""
Trade Journal Module

SQLite-based trade logging and performance tracking:
- Trade entry/exit logging with notes
- Performance analytics by strategy/symbol
- Trade history queries
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
import os


# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'trade_journal.db')


@dataclass
class JournalEntry:
    """A trade journal entry."""
    id: int = None
    symbol: str = ""
    side: str = ""  # BUY or SELL
    entry_date: str = ""
    entry_price: float = 0.0
    exit_date: str = ""
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    strategy: str = ""
    signal_type: str = ""
    stop_loss: float = None
    take_profit: float = None
    notes: str = ""
    tags: str = ""  # Comma-separated tags
    setup_quality: int = 0  # 1-5 rating
    execution_quality: int = 0  # 1-5 rating
    lessons: str = ""
    created_at: str = ""


class TradeJournal:
    """
    SQLite-based trade journal for logging and analyzing trades.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_date TEXT,
                entry_price REAL,
                exit_date TEXT,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                pnl_pct REAL,
                r_multiple REAL,
                strategy TEXT,
                signal_type TEXT,
                stop_loss REAL,
                take_profit REAL,
                notes TEXT,
                tags TEXT,
                setup_quality INTEGER,
                execution_quality INTEGER,
                lessons TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_date ON trades(entry_date)')
        
        conn.commit()
        conn.close()
    
    def add_trade(self, entry: JournalEntry) -> int:
        """
        Add a trade to the journal.
        
        Args:
            entry: JournalEntry object
        
        Returns:
            ID of the inserted trade
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                symbol, side, entry_date, entry_price, exit_date, exit_price,
                quantity, pnl, pnl_pct, r_multiple, strategy, signal_type,
                stop_loss, take_profit, notes, tags, setup_quality,
                execution_quality, lessons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.symbol, entry.side, entry.entry_date, entry.entry_price,
            entry.exit_date, entry.exit_price, entry.quantity, entry.pnl,
            entry.pnl_pct, entry.r_multiple, entry.strategy, entry.signal_type,
            entry.stop_loss, entry.take_profit, entry.notes, entry.tags,
            entry.setup_quality, entry.execution_quality, entry.lessons
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def update_trade(self, trade_id: int, updates: Dict) -> bool:
        """
        Update an existing trade.
        
        Args:
            trade_id: ID of trade to update
            updates: Dict of field: value pairs to update
        
        Returns:
            True if successful
        """
        if not updates:
            return False
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [trade_id]
        
        cursor.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", values)
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def delete_trade(self, trade_id: int) -> bool:
        """Delete a trade from the journal."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def get_trade(self, trade_id: int) -> Optional[JournalEntry]:
        """Get a single trade by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_entry(row, cursor.description)
        return None
    
    def get_all_trades(self, limit: int = 1000) -> List[JournalEntry]:
        """Get all trades, most recent first."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM trades ORDER BY entry_date DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        return [self._row_to_entry(row, description) for row in rows]
    
    def get_trades_by_symbol(self, symbol: str) -> List[JournalEntry]:
        """Get all trades for a specific symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY entry_date DESC",
            (symbol,)
        )
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        return [self._row_to_entry(row, description) for row in rows]
    
    def get_trades_by_strategy(self, strategy: str) -> List[JournalEntry]:
        """Get all trades for a specific strategy."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM trades WHERE strategy = ? ORDER BY entry_date DESC",
            (strategy,)
        )
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        return [self._row_to_entry(row, description) for row in rows]
    
    def get_trades_by_date_range(
        self, 
        start_date: str, 
        end_date: str
    ) -> List[JournalEntry]:
        """Get trades within a date range."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT * FROM trades 
               WHERE entry_date >= ? AND entry_date <= ?
               ORDER BY entry_date DESC""",
            (start_date, end_date)
        )
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        return [self._row_to_entry(row, description) for row in rows]
    
    def _row_to_entry(self, row: tuple, description) -> JournalEntry:
        """Convert database row to JournalEntry."""
        columns = [d[0] for d in description]
        data = dict(zip(columns, row))
        return JournalEntry(**data)
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        trades = self.get_all_trades()
        
        if not trades:
            return {'message': 'No trades recorded'}
        
        pnls = [t.pnl for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]
        
        total_pnl = sum(pnls)
        win_rate = len(winners) / len(trades) if trades else 0
        
        avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'avg_r_multiple': avg_r,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }
    
    def get_performance_by_strategy(self) -> pd.DataFrame:
        """Get performance breakdown by strategy."""
        trades = self.get_all_trades()
        
        if not trades:
            return pd.DataFrame()
        
        strategies = {}
        for trade in trades:
            strat = trade.strategy or 'Unknown'
            if strat not in strategies:
                strategies[strat] = []
            strategies[strat].append(trade)
        
        rows = []
        for strat, strat_trades in strategies.items():
            pnls = [t.pnl for t in strat_trades]
            winners = [t for t in strat_trades if t.pnl > 0]
            
            rows.append({
                'Strategy': strat,
                'Trades': len(strat_trades),
                'Win Rate': f"{len(winners)/len(strat_trades)*100:.1f}%",
                'Total P&L': f"${sum(pnls):,.2f}",
                'Avg P&L': f"${np.mean(pnls):,.2f}"
            })
        
        return pd.DataFrame(rows)
    
    def get_performance_by_symbol(self) -> pd.DataFrame:
        """Get performance breakdown by symbol."""
        trades = self.get_all_trades()
        
        if not trades:
            return pd.DataFrame()
        
        symbols = {}
        for trade in trades:
            sym = trade.symbol
            if sym not in symbols:
                symbols[sym] = []
            symbols[sym].append(trade)
        
        rows = []
        for sym, sym_trades in symbols.items():
            pnls = [t.pnl for t in sym_trades]
            winners = [t for t in sym_trades if t.pnl > 0]
            
            rows.append({
                'Symbol': sym,
                'Trades': len(sym_trades),
                'Win Rate': f"{len(winners)/len(sym_trades)*100:.1f}%",
                'Total P&L': f"${sum(pnls):,.2f}",
                'Avg P&L': f"${np.mean(pnls):,.2f}"
            })
        
        return pd.DataFrame(rows).sort_values('Total P&L', ascending=False)
    
    def get_monthly_performance(self) -> pd.DataFrame:
        """Get monthly P&L breakdown."""
        trades = self.get_all_trades()
        
        if not trades:
            return pd.DataFrame()
        
        monthly = {}
        for trade in trades:
            if trade.exit_date:
                month = trade.exit_date[:7]  # YYYY-MM
                if month not in monthly:
                    monthly[month] = {'pnl': 0, 'trades': 0, 'wins': 0}
                monthly[month]['pnl'] += trade.pnl
                monthly[month]['trades'] += 1
                if trade.pnl > 0:
                    monthly[month]['wins'] += 1
        
        rows = []
        for month, data in sorted(monthly.items()):
            rows.append({
                'Month': month,
                'P&L': f"${data['pnl']:,.2f}",
                'Trades': data['trades'],
                'Win Rate': f"{data['wins']/data['trades']*100:.1f}%" if data['trades'] > 0 else '0%'
            })
        
        return pd.DataFrame(rows)
    
    def export_to_csv(self, filepath: str):
        """Export all trades to CSV."""
        trades = self.get_all_trades()
        if trades:
            df = pd.DataFrame([asdict(t) for t in trades])
            df.to_csv(filepath, index=False)
    
    def import_from_csv(self, filepath: str) -> int:
        """
        Import trades from CSV.
        
        Returns:
            Number of trades imported
        """
        df = pd.read_csv(filepath)
        count = 0
        
        for _, row in df.iterrows():
            entry = JournalEntry(
                symbol=row.get('symbol', ''),
                side=row.get('side', ''),
                entry_date=str(row.get('entry_date', '')),
                entry_price=float(row.get('entry_price', 0)),
                exit_date=str(row.get('exit_date', '')),
                exit_price=float(row.get('exit_price', 0)),
                quantity=int(row.get('quantity', 0)),
                pnl=float(row.get('pnl', 0)),
                pnl_pct=float(row.get('pnl_pct', 0)),
                r_multiple=float(row.get('r_multiple', 0)),
                strategy=row.get('strategy', ''),
                signal_type=row.get('signal_type', ''),
                notes=row.get('notes', ''),
                tags=row.get('tags', '')
            )
            self.add_trade(entry)
            count += 1
        
        return count


def format_journal_summary(summary: Dict) -> pd.DataFrame:
    """Format journal summary for display."""
    rows = [
        ('Total Trades', summary.get('total_trades', 0)),
        ('Win Rate', f"{summary.get('win_rate', 0)*100:.1f}%"),
        ('Total P&L', f"${summary.get('total_pnl', 0):,.2f}"),
        ('Profit Factor', f"{summary.get('profit_factor', 0):.2f}"),
        ('Avg Winner', f"${summary.get('avg_winner', 0):,.2f}"),
        ('Avg Loser', f"${summary.get('avg_loser', 0):,.2f}"),
        ('Avg R-Multiple', f"{summary.get('avg_r_multiple', 0):.2f}R"),
        ('Best Trade', f"${summary.get('best_trade', 0):,.2f}"),
        ('Worst Trade', f"${summary.get('worst_trade', 0):,.2f}"),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Value'])
