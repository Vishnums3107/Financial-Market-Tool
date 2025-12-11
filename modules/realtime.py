"""
Real-Time Data Module

WebSocket streaming and live data updates:
- Real-time price streaming
- Live candlestick updates
- Data provider integrations
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue
import warnings


@dataclass
class Tick:
    """A single price tick."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class Candle:
    """A candlestick bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class RealtimeDataManager:
    """
    Manages real-time data streaming and updates.
    """
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.latest_prices: Dict[str, float] = {}
        self.latest_candles: Dict[str, Candle] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_queue = queue.Queue()
    
    def subscribe(self, symbol: str, callback: Callable):
        """
        Subscribe to real-time updates for a symbol.
        
        Args:
            symbol: Stock/crypto symbol
            callback: Function to call with new data
        """
        symbol = symbol.upper()
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        self.subscriptions[symbol].append(callback)
    
    def unsubscribe(self, symbol: str, callback: Callable = None):
        """
        Unsubscribe from updates.
        
        Args:
            symbol: Symbol to unsubscribe
            callback: Specific callback to remove (or all if None)
        """
        symbol = symbol.upper()
        if symbol in self.subscriptions:
            if callback:
                self.subscriptions[symbol] = [
                    cb for cb in self.subscriptions[symbol] if cb != callback
                ]
            else:
                del self.subscriptions[symbol]
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        return self.latest_prices.get(symbol.upper())
    
    def get_latest_candle(self, symbol: str) -> Optional[Candle]:
        """Get the latest candle for a symbol."""
        return self.latest_candles.get(symbol.upper())
    
    def _notify_subscribers(self, symbol: str, data: Any):
        """Notify all subscribers of new data."""
        symbol = symbol.upper()
        if symbol in self.subscriptions:
            for callback in self.subscriptions[symbol]:
                try:
                    callback(data)
                except Exception as e:
                    warnings.warn(f"Callback error for {symbol}: {e}")
    
    def process_tick(self, tick: Tick):
        """Process an incoming tick."""
        symbol = tick.symbol.upper()
        self.latest_prices[symbol] = tick.price
        
        # Update candle
        if symbol in self.latest_candles:
            candle = self.latest_candles[symbol]
            candle.high = max(candle.high, tick.price)
            candle.low = min(candle.low, tick.price)
            candle.close = tick.price
            candle.volume += tick.volume
        else:
            self.latest_candles[symbol] = Candle(
                symbol=symbol,
                timestamp=tick.timestamp,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.volume
            )
        
        # Notify subscribers
        self._notify_subscribers(symbol, tick)
    
    def simulate_realtime(
        self,
        symbol: str,
        data,
        delay: float = 1.0
    ):
        """
        Simulate real-time data from historical data.
        
        Args:
            symbol: Symbol name
            data: Historical DataFrame
            delay: Delay between ticks in seconds
        """
        import time
        
        self._running = True
        
        for idx, row in data.iterrows():
            if not self._running:
                break
            
            tick = Tick(
                symbol=symbol,
                price=row['Close'],
                volume=int(row['Volume']),
                timestamp=idx if hasattr(idx, 'strftime') else datetime.now()
            )
            
            self.process_tick(tick)
            time.sleep(delay)
    
    def stop(self):
        """Stop real-time streaming."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class YahooFinanceStream:
    """
    Yahoo Finance data polling (not true WebSocket).
    """
    
    def __init__(self, manager: RealtimeDataManager):
        self.manager = manager
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, symbols: List[str], interval: float = 5.0):
        """
        Start polling for price updates.
        
        Args:
            symbols: List of symbols to track
            interval: Polling interval in seconds
        """
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(symbols, interval)
        )
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        """Stop polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _poll_loop(self, symbols: List[str], interval: float):
        """Polling loop."""
        import time
        
        while self._running:
            for symbol in symbols:
                try:
                    price = self._get_current_price(symbol)
                    if price:
                        tick = Tick(
                            symbol=symbol,
                            price=price,
                            volume=0,
                            timestamp=datetime.now()
                        )
                        self.manager.process_tick(tick)
                except Exception as e:
                    warnings.warn(f"Error fetching {symbol}: {e}")
            
            time.sleep(interval)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return None


class BinanceWebSocket:
    """
    Binance WebSocket for crypto real-time data.
    """
    
    def __init__(self, manager: RealtimeDataManager):
        self.manager = manager
        self._ws = None
        self._running = False
    
    async def connect(self, symbols: List[str]):
        """
        Connect to Binance WebSocket.
        
        Args:
            symbols: List of crypto symbols (e.g., ['btcusdt', 'ethusdt'])
        """
        try:
            import websockets
        except ImportError:
            warnings.warn("websockets not installed. Run: pip install websockets")
            return
        
        streams = '/'.join([f"{s.lower()}@trade" for s in symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        self._running = True
        
        async with websockets.connect(url) as ws:
            self._ws = ws
            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(msg)
                    
                    if 'data' in data:
                        trade = data['data']
                        symbol = trade['s']  # Symbol
                        price = float(trade['p'])  # Price
                        qty = float(trade['q'])  # Quantity
                        
                        tick = Tick(
                            symbol=symbol,
                            price=price,
                            volume=int(qty),
                            timestamp=datetime.fromtimestamp(trade['T'] / 1000)
                        )
                        self.manager.process_tick(tick)
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    warnings.warn(f"WebSocket error: {e}")
                    break
    
    def stop(self):
        """Stop WebSocket connection."""
        self._running = False


class AlpacaStream:
    """
    Alpaca Markets WebSocket for stocks (requires API key).
    """
    
    def __init__(self, manager: RealtimeDataManager, api_key: str = "", api_secret: str = ""):
        self.manager = manager
        self.api_key = api_key
        self.api_secret = api_secret
        self._running = False
    
    async def connect(self, symbols: List[str]):
        """Connect to Alpaca stream."""
        if not self.api_key or not self.api_secret:
            warnings.warn("Alpaca API credentials required")
            return
        
        try:
            import websockets
        except ImportError:
            warnings.warn("websockets not installed")
            return
        
        url = "wss://stream.data.alpaca.markets/v2/iex"
        
        async with websockets.connect(url) as ws:
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await ws.send(json.dumps(auth_msg))
            
            # Subscribe
            sub_msg = {
                "action": "subscribe",
                "trades": symbols
            }
            await ws.send(json.dumps(sub_msg))
            
            self._running = True
            
            while self._running:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    for item in data:
                        if item.get('T') == 't':  # Trade
                            tick = Tick(
                                symbol=item['S'],
                                price=float(item['p']),
                                volume=int(item['s']),
                                timestamp=datetime.fromisoformat(item['t'].replace('Z', '+00:00'))
                            )
                            self.manager.process_tick(tick)
                
                except Exception as e:
                    warnings.warn(f"Alpaca error: {e}")
                    break
    
    def stop(self):
        """Stop streaming."""
        self._running = False


def create_realtime_manager() -> RealtimeDataManager:
    """Create a new real-time data manager."""
    return RealtimeDataManager()


def start_yahoo_polling(
    manager: RealtimeDataManager,
    symbols: List[str],
    interval: float = 5.0
) -> YahooFinanceStream:
    """
    Start Yahoo Finance polling.
    
    Args:
        manager: RealtimeDataManager instance
        symbols: Symbols to track
        interval: Polling interval
    
    Returns:
        YahooFinanceStream instance
    """
    stream = YahooFinanceStream(manager)
    stream.start(symbols, interval)
    return stream


def format_tick_display(tick: Tick) -> Dict:
    """Format tick for display."""
    return {
        'Symbol': tick.symbol,
        'Price': f"${tick.price:.2f}",
        'Volume': f"{tick.volume:,}",
        'Time': tick.timestamp.strftime('%H:%M:%S') if hasattr(tick.timestamp, 'strftime') else str(tick.timestamp)
    }
