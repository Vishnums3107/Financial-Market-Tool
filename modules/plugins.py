"""
Plugin System Module

Extensibility framework for custom indicators and strategies:
- Custom indicator formula parser
- External strategy module loader
- Plugin registration and management
"""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import warnings


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    type: str  # 'indicator' or 'strategy'
    description: str
    author: str = "Unknown"
    version: str = "1.0.0"
    enabled: bool = True


@dataclass
class CustomIndicator:
    """A custom indicator definition."""
    name: str
    formula: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class FormulaParser:
    """
    Parse and evaluate custom indicator formulas.
    
    Supported functions:
    - SMA(period): Simple Moving Average
    - EMA(period): Exponential Moving Average
    - RSI(period): Relative Strength Index
    - CLOSE, OPEN, HIGH, LOW, VOLUME: Price data
    - Basic math: +, -, *, /, **, ()
    - Comparisons: >, <, >=, <=, ==
    - Conditionals: IF(condition, true_val, false_val)
    """
    
    ALLOWED_FUNCTIONS = {
        'SMA': lambda data, period: data['Close'].rolling(int(period)).mean(),
        'EMA': lambda data, period: data['Close'].ewm(span=int(period), adjust=False).mean(),
        'RSI': lambda data, period: _calc_rsi(data, int(period)),
        'ATR': lambda data, period: _calc_atr(data, int(period)),
        'STDEV': lambda data, period: data['Close'].rolling(int(period)).std(),
        'MIN': lambda data, period: data['Close'].rolling(int(period)).min(),
        'MAX': lambda data, period: data['Close'].rolling(int(period)).max(),
        'SUM': lambda data, period: data['Close'].rolling(int(period)).sum(),
        'ABS': lambda data, val: np.abs(val),
        'SQRT': lambda data, val: np.sqrt(val),
        'LOG': lambda data, val: np.log(val),
        'CROSSOVER': lambda data, a, b: (a > b) & (a.shift(1) <= b.shift(1)),
        'CROSSUNDER': lambda data, a, b: (a < b) & (a.shift(1) >= b.shift(1)),
    }
    
    PRICE_VARS = {
        'CLOSE': lambda data: data['Close'],
        'OPEN': lambda data: data['Open'],
        'HIGH': lambda data: data['High'],
        'LOW': lambda data: data['Low'],
        'VOLUME': lambda data: data['Volume'],
        'HL2': lambda data: (data['High'] + data['Low']) / 2,
        'HLC3': lambda data: (data['High'] + data['Low'] + data['Close']) / 3,
        'OHLC4': lambda data: (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4,
    }
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._cache = {}
    
    def evaluate(self, formula: str) -> pd.Series:
        """
        Evaluate a formula string.
        
        Args:
            formula: Formula string like "SMA(20) - SMA(50)"
        
        Returns:
            Pandas Series with result
        """
        try:
            # Replace price variables
            expr = formula.upper()
            for var, func in self.PRICE_VARS.items():
                if var in expr:
                    self._cache[var] = func(self.data)
            
            # Parse and evaluate
            result = self._parse_expression(expr)
            return result
        except Exception as e:
            warnings.warn(f"Formula evaluation error: {e}")
            return pd.Series(np.nan, index=self.data.index)
    
    def _parse_expression(self, expr: str) -> pd.Series:
        """Parse and evaluate expression."""
        expr = expr.strip()
        
        # Check for function calls
        for func_name, func in self.ALLOWED_FUNCTIONS.items():
            if expr.startswith(func_name + '('):
                # Extract arguments
                paren_start = expr.index('(')
                paren_end = self._find_matching_paren(expr, paren_start)
                args_str = expr[paren_start + 1:paren_end]
                
                # Parse arguments
                args = self._parse_args(args_str)
                
                # Evaluate function
                return func(self.data, *args)
        
        # Check for price variables
        if expr in self._cache:
            return self._cache[expr]
        if expr in self.PRICE_VARS:
            return self.PRICE_VARS[expr](self.data)
        
        # Check for number
        try:
            return float(expr)
        except ValueError:
            pass
        
        # Handle operators
        for op in ['+', '-', '*', '/', '**']:
            if op in expr:
                # Find operator not inside parentheses
                level = 0
                for i, c in enumerate(expr):
                    if c == '(':
                        level += 1
                    elif c == ')':
                        level -= 1
                    elif c == op[0] and level == 0 and i > 0:
                        left = self._parse_expression(expr[:i])
                        right = self._parse_expression(expr[i + len(op):])
                        
                        if op == '+':
                            return left + right
                        elif op == '-':
                            return left - right
                        elif op == '*':
                            return left * right
                        elif op == '/':
                            return left / right
                        elif op == '**':
                            return left ** right
        
        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse_expression(expr[1:-1])
        
        raise ValueError(f"Cannot parse expression: {expr}")
    
    def _find_matching_paren(self, s: str, start: int) -> int:
        """Find matching closing parenthesis."""
        level = 1
        for i in range(start + 1, len(s)):
            if s[i] == '(':
                level += 1
            elif s[i] == ')':
                level -= 1
                if level == 0:
                    return i
        return len(s)
    
    def _parse_args(self, args_str: str) -> List:
        """Parse function arguments."""
        args = []
        current = ""
        level = 0
        
        for c in args_str + ',':
            if c == '(':
                level += 1
                current += c
            elif c == ')':
                level -= 1
                current += c
            elif c == ',' and level == 0:
                if current.strip():
                    # Try to parse as number first
                    try:
                        args.append(float(current.strip()))
                    except ValueError:
                        # Parse as expression
                        args.append(self._parse_expression(current.strip()))
                current = ""
            else:
                current += c
        
        return args


def _calc_rsi(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate RSI."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate ATR."""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


class PluginManager:
    """
    Manages loading and execution of plugins.
    """
    
    def __init__(self, plugin_dir: str = None):
        self.plugin_dir = plugin_dir or os.path.join(
            os.path.dirname(__file__), '..', 'plugins'
        )
        self.indicators: Dict[str, CustomIndicator] = {}
        self.strategies: Dict[str, Callable] = {}
        self.plugins: Dict[str, PluginInfo] = {}
        
        # Ensure plugin directory exists
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
    
    def register_indicator(self, indicator: CustomIndicator):
        """Register a custom indicator."""
        self.indicators[indicator.name] = indicator
    
    def register_strategy(self, name: str, func: Callable, description: str = ""):
        """Register a custom strategy function."""
        self.strategies[name] = func
        self.plugins[name] = PluginInfo(
            name=name,
            type='strategy',
            description=description
        )
    
    def load_plugins_from_dir(self):
        """Load all plugins from the plugin directory."""
        if not os.path.exists(self.plugin_dir):
            return
        
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                self._load_plugin_file(os.path.join(self.plugin_dir, filename))
    
    def _load_plugin_file(self, filepath: str):
        """Load a single plugin file."""
        try:
            spec = importlib.util.spec_from_file_location("plugin", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for register function
            if hasattr(module, 'register'):
                module.register(self)
            
            # Look for INDICATOR or STRATEGY definitions
            if hasattr(module, 'INDICATOR'):
                self.register_indicator(module.INDICATOR)
            
            if hasattr(module, 'strategy') and callable(module.strategy):
                name = os.path.basename(filepath).replace('.py', '')
                desc = getattr(module, '__doc__', '') or ''
                self.register_strategy(name, module.strategy, desc)
                
        except Exception as e:
            warnings.warn(f"Failed to load plugin {filepath}: {e}")
    
    def apply_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        **params
    ) -> pd.DataFrame:
        """
        Apply a custom indicator to data.
        
        Args:
            data: OHLCV DataFrame
            indicator_name: Name of registered indicator
            params: Override parameters
        
        Returns:
            DataFrame with indicator added
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        indicator = self.indicators[indicator_name]
        formula = indicator.formula
        
        # Apply parameter substitutions
        merged_params = {**indicator.parameters, **params}
        for key, value in merged_params.items():
            formula = formula.replace(f'{{{key}}}', str(value))
        
        # Evaluate formula
        parser = FormulaParser(data)
        result = parser.evaluate(formula)
        
        data = data.copy()
        data[indicator_name] = result
        
        return data
    
    def run_strategy(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        **params
    ) -> pd.DataFrame:
        """
        Run a custom strategy.
        
        Args:
            data: OHLCV DataFrame
            strategy_name: Name of registered strategy
            params: Strategy parameters
        
        Returns:
            DataFrame with signals
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return self.strategies[strategy_name](data, **params)
    
    def list_indicators(self) -> List[str]:
        """List all registered indicators."""
        return list(self.indicators.keys())
    
    def list_strategies(self) -> List[str]:
        """List all registered strategies."""
        return list(self.strategies.keys())
    
    def get_plugin_info(self) -> pd.DataFrame:
        """Get info about all plugins."""
        rows = []
        
        for name, indicator in self.indicators.items():
            rows.append({
                'Name': name,
                'Type': 'Indicator',
                'Description': indicator.description,
                'Formula': indicator.formula[:50] + '...' if len(indicator.formula) > 50 else indicator.formula
            })
        
        for name, info in self.plugins.items():
            if info.type == 'strategy':
                rows.append({
                    'Name': name,
                    'Type': 'Strategy',
                    'Description': info.description,
                    'Formula': 'N/A'
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['Name', 'Type', 'Description', 'Formula']
        )


# Example built-in custom indicators
BUILTIN_INDICATORS = [
    CustomIndicator(
        name='MACD_HIST',
        formula='EMA(12) - EMA(26)',
        description='MACD Histogram'
    ),
    CustomIndicator(
        name='PRICE_VS_SMA',
        formula='(CLOSE - SMA({period})) / SMA({period}) * 100',
        description='Price vs SMA percentage',
        parameters={'period': 20}
    ),
    CustomIndicator(
        name='ATR_PERCENT',
        formula='ATR({period}) / CLOSE * 100',
        description='ATR as percentage of price',
        parameters={'period': 14}
    ),
    CustomIndicator(
        name='VOLATILITY',
        formula='STDEV({period}) / CLOSE * 100',
        description='Volatility (rolling std)',
        parameters={'period': 20}
    ),
    CustomIndicator(
        name='MOMENTUM',
        formula='CLOSE - CLOSE',  # This is a placeholder
        description='Price momentum'
    ),
]


def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager."""
    manager = PluginManager()
    
    # Register built-in indicators
    for indicator in BUILTIN_INDICATORS:
        manager.register_indicator(indicator)
    
    # Load external plugins
    manager.load_plugins_from_dir()
    
    return manager


def create_indicator_from_formula(
    name: str,
    formula: str,
    description: str = "",
    **params
) -> CustomIndicator:
    """
    Create a custom indicator from a formula string.
    
    Example:
        indicator = create_indicator_from_formula(
            'MY_INDICATOR',
            'SMA(20) - SMA(50)',
            'My custom crossover indicator'
        )
    """
    return CustomIndicator(
        name=name,
        formula=formula,
        description=description,
        parameters=params
    )
