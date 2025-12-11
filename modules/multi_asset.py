"""
Multi-Asset Support Module

Support for multiple asset classes:
- Cryptocurrency (via yfinance or Binance)
- Forex pairs with session overlays
- Commodities
- Cross-asset correlation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


# Asset class definitions
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'Ripple',
    'ADA-USD': 'Cardano',
    'SOL-USD': 'Solana',
    'DOGE-USD': 'Dogecoin',
    'DOT-USD': 'Polkadot',
    'AVAX-USD': 'Avalanche',
    'MATIC-USD': 'Polygon'
}

FOREX_PAIRS = {
    'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD',
    'USDJPY=X': 'USD/JPY',
    'AUDUSD=X': 'AUD/USD',
    'USDCAD=X': 'USD/CAD',
    'USDCHF=X': 'USD/CHF',
    'NZDUSD=X': 'NZD/USD',
    'EURGBP=X': 'EUR/GBP',
    'EURJPY=X': 'EUR/JPY',
    'GBPJPY=X': 'GBP/JPY'
}

COMMODITY_SYMBOLS = {
    'GC=F': 'Gold Futures',
    'SI=F': 'Silver Futures',
    'CL=F': 'Crude Oil WTI',
    'BZ=F': 'Brent Crude',
    'NG=F': 'Natural Gas',
    'HG=F': 'Copper',
    'PL=F': 'Platinum',
    'ZC=F': 'Corn',
    'ZW=F': 'Wheat',
    'ZS=F': 'Soybeans'
}

# Forex session times (UTC)
FOREX_SESSIONS = {
    'Sydney': {'start': 21, 'end': 6},
    'Tokyo': {'start': 0, 'end': 9},
    'London': {'start': 7, 'end': 16},
    'New York': {'start': 12, 'end': 21}
}


@dataclass
class AssetInfo:
    """Information about an asset."""
    symbol: str
    name: str
    asset_class: str
    exchange: str = ""
    currency: str = "USD"


def get_asset_class(symbol: str) -> str:
    """Determine asset class from symbol."""
    if symbol in CRYPTO_SYMBOLS or '-USD' in symbol:
        return 'Crypto'
    elif symbol in FOREX_PAIRS or '=X' in symbol:
        return 'Forex'
    elif symbol in COMMODITY_SYMBOLS or '=F' in symbol:
        return 'Commodity'
    else:
        return 'Stock'


def get_all_assets() -> Dict[str, Dict]:
    """Get all available assets by category."""
    return {
        'Crypto': CRYPTO_SYMBOLS,
        'Forex': FOREX_PAIRS,
        'Commodities': COMMODITY_SYMBOLS
    }


def get_crypto_list() -> List[str]:
    """Get list of crypto symbols."""
    return list(CRYPTO_SYMBOLS.keys())


def get_forex_list() -> List[str]:
    """Get list of forex pairs."""
    return list(FOREX_PAIRS.keys())


def get_commodity_list() -> List[str]:
    """Get list of commodity symbols."""
    return list(COMMODITY_SYMBOLS.keys())


def fetch_multi_asset_data(
    symbols: List[str],
    start: str,
    end: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple assets.
    
    Args:
        symbols: List of symbols
        start: Start date
        end: End date
    
    Returns:
        Dict of {symbol: DataFrame}
    """
    import yfinance as yf
    
    result = {}
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start, end=end, progress=False)
            if not data.empty:
                # Flatten multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                result[symbol] = data
        except Exception as e:
            warnings.warn(f"Failed to fetch {symbol}: {e}")
    
    return result


def get_current_forex_session(utc_hour: int = None) -> List[str]:
    """
    Get currently active forex sessions.
    
    Args:
        utc_hour: Current UTC hour (default: now)
    
    Returns:
        List of active session names
    """
    if utc_hour is None:
        utc_hour = datetime.utcnow().hour
    
    active = []
    
    for session, times in FOREX_SESSIONS.items():
        start = times['start']
        end = times['end']
        
        if start < end:
            if start <= utc_hour < end:
                active.append(session)
        else:  # Wraps around midnight
            if utc_hour >= start or utc_hour < end:
                active.append(session)
    
    return active


def add_session_overlay(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add forex session information to data.
    
    Args:
        data: OHLCV DataFrame with DatetimeIndex
    
    Returns:
        DataFrame with Session column
    """
    data = data.copy()
    
    def get_session(dt):
        if hasattr(dt, 'hour'):
            hour = dt.hour
        else:
            hour = dt.to_pydatetime().hour
        
        sessions = get_current_forex_session(hour)
        return ', '.join(sessions) if sessions else 'Off-market'
    
    data['Session'] = data.index.map(get_session)
    
    return data


def calculate_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    method: str = 'pearson',
    window: int = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        data_dict: Dict of {symbol: DataFrame}
        method: 'pearson', 'spearman', or 'kendall'
        window: Rolling window for rolling correlation
    
    Returns:
        Correlation matrix
    """
    # Extract returns
    returns = pd.DataFrame()
    
    for symbol, data in data_dict.items():
        if 'Close' in data.columns:
            returns[symbol] = data['Close'].pct_change()
    
    returns = returns.dropna()
    
    if returns.empty or len(returns.columns) < 2:
        return pd.DataFrame()
    
    if window:
        # Return rolling correlation of last two columns
        return returns.iloc[:, 0].rolling(window).corr(returns.iloc[:, 1])
    
    return returns.corr(method=method)


def analyze_intermarket_relationships(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame = None,
    dxy_data: pd.DataFrame = None,
    gold_data: pd.DataFrame = None
) -> Dict:
    """
    Analyze intermarket relationships.
    
    Args:
        spy_data: S&P 500 data
        vix_data: VIX data
        dxy_data: Dollar index data
        gold_data: Gold data
    
    Returns:
        Dict with intermarket analysis
    """
    results = {}
    
    # SPY trend
    if not spy_data.empty:
        spy_sma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
        spy_current = spy_data['Close'].iloc[-1]
        results['spy_trend'] = 'Bullish' if spy_current > spy_sma50 else 'Bearish'
        results['spy_vs_sma50'] = ((spy_current / spy_sma50) - 1) * 100
    
    # VIX analysis (fear gauge)
    if vix_data is not None and not vix_data.empty:
        vix_current = vix_data['Close'].iloc[-1]
        vix_avg = vix_data['Close'].rolling(20).mean().iloc[-1]
        
        if vix_current < 15:
            results['vix_regime'] = 'Low Volatility'
        elif vix_current < 25:
            results['vix_regime'] = 'Normal'
        elif vix_current < 35:
            results['vix_regime'] = 'Elevated'
        else:
            results['vix_regime'] = 'High Fear'
        
        results['vix_level'] = vix_current
    
    # Dollar analysis
    if dxy_data is not None and not dxy_data.empty:
        dxy_sma20 = dxy_data['Close'].rolling(20).mean().iloc[-1]
        dxy_current = dxy_data['Close'].iloc[-1]
        results['dollar_trend'] = 'Strong' if dxy_current > dxy_sma20 else 'Weak'
    
    # Gold analysis (safe haven)
    if gold_data is not None and not gold_data.empty:
        gold_sma50 = gold_data['Close'].rolling(50).mean().iloc[-1]
        gold_current = gold_data['Close'].iloc[-1]
        results['gold_trend'] = 'Bullish' if gold_current > gold_sma50 else 'Bearish'
    
    # Risk regime determination
    spy_bullish = results.get('spy_trend') == 'Bullish'
    vix_low = results.get('vix_level', 20) < 20
    
    if spy_bullish and vix_low:
        results['risk_regime'] = 'Risk-On'
    elif not spy_bullish and not vix_low:
        results['risk_regime'] = 'Risk-Off'
    else:
        results['risk_regime'] = 'Mixed'
    
    return results


def get_crypto_metrics(symbol: str) -> Dict:
    """
    Get basic crypto metrics (via yfinance).
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD')
    
    Returns:
        Dict with crypto metrics
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('name', symbol),
            'price': info.get('regularMarketPrice', 0),
            'market_cap': info.get('marketCap', 0),
            'volume_24h': info.get('volume24Hr', info.get('volume', 0)),
            'change_24h': info.get('regularMarketChangePercent', 0),
            'high_24h': info.get('dayHigh', 0),
            'low_24h': info.get('dayLow', 0),
            'circulating_supply': info.get('circulatingSupply', 0)
        }
    except Exception as e:
        return {'error': str(e)}


def compare_assets(
    symbols: List[str],
    start: str,
    end: str,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compare performance of multiple assets.
    
    Args:
        symbols: List of symbols to compare
        start: Start date
        end: End date
        normalize: Whether to normalize to 100
    
    Returns:
        DataFrame with performance comparison
    """
    data_dict = fetch_multi_asset_data(symbols, start, end)
    
    if not data_dict:
        return pd.DataFrame()
    
    comparison = pd.DataFrame()
    
    for symbol, data in data_dict.items():
        if 'Close' in data.columns:
            prices = data['Close']
            if normalize:
                prices = (prices / prices.iloc[0]) * 100
            comparison[symbol] = prices
    
    return comparison


def format_correlation_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Format correlation matrix for display with colors."""
    if corr_matrix.empty:
        return corr_matrix
    
    def format_corr(val):
        if val == 1.0:
            return "1.00"
        elif val > 0.7:
            return f"🟢 {val:.2f}"
        elif val > 0.3:
            return f"🟡 {val:.2f}"
        elif val > -0.3:
            return f"⚪ {val:.2f}"
        elif val > -0.7:
            return f"🟠 {val:.2f}"
        else:
            return f"🔴 {val:.2f}"
    
    return corr_matrix.applymap(format_corr)


def format_intermarket_summary(analysis: Dict) -> pd.DataFrame:
    """Format intermarket analysis for display."""
    rows = []
    
    if 'spy_trend' in analysis:
        emoji = '🟢' if analysis['spy_trend'] == 'Bullish' else '🔴'
        rows.append(('S&P 500 Trend', f"{emoji} {analysis['spy_trend']}"))
    
    if 'vix_regime' in analysis:
        rows.append(('VIX Regime', f"{analysis['vix_regime']} ({analysis.get('vix_level', 0):.1f})"))
    
    if 'dollar_trend' in analysis:
        rows.append(('Dollar', analysis['dollar_trend']))
    
    if 'gold_trend' in analysis:
        rows.append(('Gold', analysis['gold_trend']))
    
    if 'risk_regime' in analysis:
        emoji = '🟢' if analysis['risk_regime'] == 'Risk-On' else '🔴' if analysis['risk_regime'] == 'Risk-Off' else '🟡'
        rows.append(('Risk Regime', f"{emoji} {analysis['risk_regime']}"))
    
    return pd.DataFrame(rows, columns=['Indicator', 'Status'])
