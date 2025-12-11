"""
Alerts Module

Alert system for trading notifications:
- Email notifications (SMTP)
- Telegram bot integration
- Webhook notifications
- Alert condition management
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import warnings


# Alert database path
ALERT_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'alerts.db')


class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CROSS_LEVEL = "price_cross_level"
    PATTERN_DETECTED = "pattern_detected"
    SIGNAL_GENERATED = "signal_generated"
    VOLUME_SPIKE = "volume_spike"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"


class AlertStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DISABLED = "disabled"


@dataclass
class Alert:
    """Alert definition."""
    id: int = None
    symbol: str = ""
    alert_type: str = ""
    condition_value: float = 0.0
    message: str = ""
    status: str = "active"
    notification_method: str = "all"  # email, telegram, webhook, all
    created_at: str = ""
    triggered_at: str = ""
    expires_at: str = ""


class AlertManager:
    """
    Manages alert creation, checking, and notifications.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ALERT_DB_PATH
        self._ensure_db_dir()
        self._init_db()
        
        # Notification settings
        self.email_config = {}
        self.telegram_config = {}
        self.webhook_config = {}
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                condition_value REAL,
                message TEXT,
                status TEXT DEFAULT 'active',
                notification_method TEXT DEFAULT 'all',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                triggered_at TEXT,
                expires_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER,
                triggered_at TEXT,
                triggered_price REAL,
                notification_sent INTEGER DEFAULT 0,
                FOREIGN KEY (alert_id) REFERENCES alerts(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_email: str
    ):
        """Configure email notifications."""
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_email': from_email,
            'to_email': to_email
        }
    
    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram notifications."""
        self.telegram_config = {
            'bot_token': bot_token,
            'chat_id': chat_id
        }
    
    def configure_webhook(self, url: str, headers: Dict = None):
        """Configure webhook notifications."""
        self.webhook_config = {
            'url': url,
            'headers': headers or {}
        }
    
    def create_alert(
        self,
        symbol: str,
        alert_type: AlertType,
        condition_value: float,
        message: str = "",
        notification_method: str = "all",
        expires_at: str = None
    ) -> int:
        """
        Create a new alert.
        
        Args:
            symbol: Stock symbol
            alert_type: Type of alert
            condition_value: Value for condition check
            message: Custom message
            notification_method: How to notify (email, telegram, webhook, all)
            expires_at: Optional expiration datetime
        
        Returns:
            Alert ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (symbol, alert_type, condition_value, message, 
                              notification_method, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, alert_type.value, condition_value, message, 
              notification_method, expires_at))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return alert_id
    
    def get_active_alerts(self, symbol: str = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute(
                "SELECT * FROM alerts WHERE status = 'active' AND symbol = ?",
                (symbol,)
            )
        else:
            cursor.execute("SELECT * FROM alerts WHERE status = 'active'")
        
        rows = cursor.fetchall()
        description = cursor.description
        conn.close()
        
        return [self._row_to_alert(row, description) for row in rows]
    
    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    
    def check_alerts(self, symbol: str, current_price: float, data: dict = None) -> List[Alert]:
        """
        Check if any alerts should be triggered.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            data: Additional data (RSI, volume, etc.)
        
        Returns:
            List of triggered alerts
        """
        active_alerts = self.get_active_alerts(symbol)
        triggered = []
        
        for alert in active_alerts:
            should_trigger = False
            
            if alert.alert_type == AlertType.PRICE_ABOVE.value:
                should_trigger = current_price > alert.condition_value
            
            elif alert.alert_type == AlertType.PRICE_BELOW.value:
                should_trigger = current_price < alert.condition_value
            
            elif alert.alert_type == AlertType.RSI_OVERBOUGHT.value and data:
                rsi = data.get('RSI', 50)
                should_trigger = rsi > alert.condition_value
            
            elif alert.alert_type == AlertType.RSI_OVERSOLD.value and data:
                rsi = data.get('RSI', 50)
                should_trigger = rsi < alert.condition_value
            
            elif alert.alert_type == AlertType.VOLUME_SPIKE.value and data:
                volume_ratio = data.get('volume_ratio', 1.0)
                should_trigger = volume_ratio > alert.condition_value
            
            if should_trigger:
                self._trigger_alert(alert, current_price)
                triggered.append(alert)
        
        return triggered
    
    def _trigger_alert(self, alert: Alert, current_price: float):
        """Trigger an alert and send notifications."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        triggered_at = datetime.now().isoformat()
        
        # Update alert status
        cursor.execute(
            "UPDATE alerts SET status = 'triggered', triggered_at = ? WHERE id = ?",
            (triggered_at, alert.id)
        )
        
        # Log to history
        cursor.execute(
            "INSERT INTO alert_history (alert_id, triggered_at, triggered_price) VALUES (?, ?, ?)",
            (alert.id, triggered_at, current_price)
        )
        
        conn.commit()
        conn.close()
        
        # Send notifications
        message = alert.message or f"Alert triggered for {alert.symbol}: {alert.alert_type} at ${current_price:.2f}"
        self._send_notifications(alert.notification_method, message)
    
    def _send_notifications(self, method: str, message: str):
        """Send notifications based on configured methods."""
        methods = [method] if method != 'all' else ['email', 'telegram', 'webhook']
        
        for m in methods:
            try:
                if m == 'email' and self.email_config:
                    self._send_email(message)
                elif m == 'telegram' and self.telegram_config:
                    self._send_telegram(message)
                elif m == 'webhook' and self.webhook_config:
                    self._send_webhook(message)
            except Exception as e:
                warnings.warn(f"Failed to send {m} notification: {e}")
    
    def _send_email(self, message: str):
        """Send email notification."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            msg = MIMEText(message)
            msg['Subject'] = 'Stock Alert'
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            
            with smtplib.SMTP(self.email_config['smtp_server'], 
                            self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], 
                           self.email_config['password'])
                server.send_message(msg)
        except ImportError:
            warnings.warn("smtplib not available")
    
    def _send_telegram(self, message: str):
        """Send Telegram notification."""
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            payload = {
                'chat_id': self.telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            requests.post(url, json=payload)
        except ImportError:
            warnings.warn("requests not available")
    
    def _send_webhook(self, message: str):
        """Send webhook notification."""
        try:
            import requests
            
            payload = {
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config.get('headers', {})
            )
        except ImportError:
            warnings.warn("requests not available")
    
    def _row_to_alert(self, row: tuple, description) -> Alert:
        """Convert database row to Alert object."""
        columns = [d[0] for d in description]
        data = dict(zip(columns, row))
        return Alert(**data)


def create_price_alert(
    symbol: str,
    price: float,
    direction: str = "above",
    message: str = None
) -> Alert:
    """Helper function to create a price alert."""
    alert_type = AlertType.PRICE_ABOVE if direction == "above" else AlertType.PRICE_BELOW
    default_message = f"{symbol} price {'above' if direction == 'above' else 'below'} ${price:.2f}"
    
    return Alert(
        symbol=symbol,
        alert_type=alert_type.value,
        condition_value=price,
        message=message or default_message,
        status=AlertStatus.ACTIVE.value
    )


def format_alerts_table(alerts: List[Alert]) -> 'pd.DataFrame':
    """Format alerts for display."""
    import pandas as pd
    
    if not alerts:
        return pd.DataFrame(columns=['Symbol', 'Type', 'Value', 'Status'])
    
    rows = [{
        'Symbol': a.symbol,
        'Type': a.alert_type,
        'Value': f"${a.condition_value:.2f}",
        'Status': a.status,
        'Created': a.created_at[:10] if a.created_at else 'N/A'
    } for a in alerts]
    
    return pd.DataFrame(rows)
