import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

def plot_stock_dashboard(data, ticker, indicators=[]):
    """
    Create an interactive stock dashboard with multiple subplots.
    
    Args:
        data (pd.DataFrame): Stock data with indicators
        ticker (str): Stock ticker symbol
        indicators (list): List of indicators to display
    
    Returns:
        plotly.graph_objects.Figure: Interactive dashboard figure
    """
    # Determine number of rows based on indicators
    num_rows = 2  # Price and Volume are always included
    if 'RSI' in data.columns or any('RSI' in ind for ind in indicators):
        num_rows += 1
    if 'MACD' in data.columns or any('MACD' in ind for ind in indicators):
        num_rows += 1
    if any('Stochastic' in str(data.columns) for _ in [1]) or '%K' in data.columns:
        num_rows += 1
    
    # Create subplot titles
    subplot_titles = ["Price & Indicators", "Volume"]
    row_heights = [0.5, 0.15]
    
    if 'RSI' in data.columns or any('RSI' in ind for ind in indicators):
        subplot_titles.append("RSI")
        row_heights.append(0.15)
    if 'MACD' in data.columns or any('MACD' in ind for ind in indicators):
        subplot_titles.append("MACD")
        row_heights.append(0.2)
    if '%K' in data.columns:
        subplot_titles.append("Stochastic")
        row_heights.append(0.15)
    
    # Normalize row heights
    row_heights = [h/sum(row_heights) for h in row_heights]
    
    fig = make_subplots(
        rows=num_rows, cols=1, 
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f"{ticker} Price",
            showlegend=False
        ), 
        row=1, col=1
    )
    
    # Moving Averages
    ma_colors = ['#FFA500', '#32CD32', '#FF69B4', '#00CED1', '#FFD700']
    color_idx = 0
    
    for ind in indicators:
        if ind in data.columns and ('SMA' in ind or 'EMA' in ind):
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data[ind], 
                    name=ind,
                    line=dict(color=ma_colors[color_idx % len(ma_colors)], width=2)
                ), 
                row=1, col=1
            )
            color_idx += 1
    
    # Bollinger Bands
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['BB_Upper'], 
                name='BB Upper',
                line=dict(color='rgba(173,216,230,0.5)', width=1)
            ), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['BB_Lower'], 
                name='BB Lower',
                line=dict(color='rgba(173,216,230,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173,216,230,0.1)'
            ), 
            row=1, col=1
        )

    # Volume
    colors = ['red' if close < open else 'green' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index, 
            y=data['Volume'], 
            name="Volume",
            marker_color=colors,
            showlegend=False
        ), 
        row=2, col=1
    )

    current_row = 3

    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['RSI'], 
                name="RSI",
                line=dict(color='orange', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", row=current_row, col=1, 
                      line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", row=current_row, col=1, 
                      line_color="green", annotation_text="Oversold (30)")
        current_row += 1

    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['MACD'], 
                name="MACD",
                line=dict(color='blue', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['Signal'], 
                name="Signal",
                line=dict(color='red', width=2)
            ), 
            row=current_row, col=1
        )
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index, 
                y=data['Histogram'], 
                name="MACD Histogram",
                marker_color=colors,
                opacity=0.7
            ), 
            row=current_row, col=1
        )
        current_row += 1
    
    # Stochastic Oscillator
    if '%K' in data.columns and '%D' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['%K'], 
                name="%K",
                line=dict(color='blue', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['%D'], 
                name="%D",
                line=dict(color='red', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_hline(y=80, line_dash="dash", row=current_row, col=1, 
                      line_color="red", annotation_text="Overbought (80)")
        fig.add_hline(y=20, line_dash="dash", row=current_row, col=1, 
                      line_color="green", annotation_text="Oversold (20)")

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Stock Analysis Dashboard",
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    # Update all x-axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def plot_comparison_chart(data_dict, tickers):
    """
    Create a comparison chart for multiple stocks.
    
    Args:
        data_dict (dict): Dictionary with ticker as key and data as value
        tickers (list): List of ticker symbols
    
    Returns:
        plotly.graph_objects.Figure: Comparison chart
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, ticker in enumerate(tickers):
        if ticker in data_dict and not data_dict[ticker].empty:
            # Normalize prices to percentage change from first day
            normalized_prices = ((data_dict[ticker]['Close'] / data_dict[ticker]['Close'].iloc[0]) - 1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=data_dict[ticker].index,
                    y=normalized_prices,
                    name=f"{ticker}",
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
    
    fig.update_layout(
        title="Stock Price Comparison (Normalized %)",
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        height=600,
        yaxis_title="Percentage Change (%)",
        xaxis_title="Date"
    )
    
    return fig

def create_summary_metrics_table(data, ticker):
    """
    Create a summary table with key metrics.
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: Summary metrics
    """
    if data.empty:
        return pd.DataFrame()
    
    current_price = float(data['Close'].iloc[-1])
    open_price = float(data['Open'].iloc[-1])
    high_52w = float(data['High'].rolling(252).max().iloc[-1])
    low_52w = float(data['Low'].rolling(252).min().iloc[-1])
    avg_volume = float(data['Volume'].rolling(30).mean().iloc[-1])
    
    metrics = {
        'Metric': [
            'Current Price', 'Daily Change', 'Daily Change %', 
            '52W High', '52W Low', '30D Avg Volume'
        ],
        'Value': [
            f"${current_price:.2f}",
            f"${current_price - open_price:.2f}",
            f"{((current_price - open_price) / open_price) * 100:.2f}%",
            f"${high_52w:.2f}",
            f"${low_52w:.2f}",
            f"{avg_volume:,.0f}"
        ]
    }
    
    return pd.DataFrame(metrics)


def plot_support_resistance_zones(fig, data, sr_data, row=1, col=1):
    """
    Add support and resistance zones to chart.
    
    Args:
        fig: Plotly figure
        data (pd.DataFrame): Stock data
        sr_data (dict): Support/resistance data
        row (int): Subplot row
        col (int): Subplot column
    
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    import plotly.graph_objects as go
    
    support_levels = sr_data.get('support_levels', [])
    resistance_levels = sr_data.get('resistance_levels', [])
    
    # Add support zones (green)
    for zone in support_levels[:5]:  # Limit to 5 zones
        level = zone['level']
        zone_range = zone['range']
        strength = zone['strength']
        opacity = min(0.3 + (strength * 0.1), 0.6)
        
        fig.add_hrect(
            y0=zone_range[0], y1=zone_range[1],
            fillcolor="rgba(0, 255, 0, {})".format(opacity * 0.3),
            line=dict(color="rgba(0, 255, 0, 0.5)", width=1),
            row=row, col=col,
            annotation_text=f"S ({strength} touches)",
            annotation_position="right"
        )
    
    # Add resistance zones (red)
    for zone in resistance_levels[:5]:
        level = zone['level']
        zone_range = zone['range']
        strength = zone['strength']
        opacity = min(0.3 + (strength * 0.1), 0.6)
        
        fig.add_hrect(
            y0=zone_range[0], y1=zone_range[1],
            fillcolor="rgba(255, 0, 0, {})".format(opacity * 0.3),
            line=dict(color="rgba(255, 0, 0, 0.5)", width=1),
            row=row, col=col,
            annotation_text=f"R ({strength} touches)",
            annotation_position="right"
        )
    
    return fig


def plot_trend_structure(fig, data, row=1, col=1):
    """
    Annotate trend structure (swing highs/lows) on chart.
    
    Args:
        fig: Plotly figure
        data (pd.DataFrame): Stock data with swing point columns
        row (int): Subplot row
        col (int): Subplot column
    
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    import plotly.graph_objects as go
    
    if 'SwingType' not in data.columns:
        return fig
    
    # Filter swing points with labels
    swing_points = data[data['SwingType'] != '']
    
    # Add markers for swing highs
    hh_points = swing_points[swing_points['SwingType'] == 'HH']
    lh_points = swing_points[swing_points['SwingType'] == 'LH']
    hl_points = swing_points[swing_points['SwingType'] == 'HL']
    ll_points = swing_points[swing_points['SwingType'] == 'LL']
    
    # Higher Highs (green triangle up)
    if not hh_points.empty:
        fig.add_trace(
            go.Scatter(
                x=hh_points.index,
                y=hh_points['High'] * 1.01,
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=12, color='#00ff00'),
                text=['HH'] * len(hh_points),
                textposition='top center',
                textfont=dict(color='#00ff00', size=10),
                name='Higher Highs',
                showlegend=True
            ),
            row=row, col=col
        )
    
    # Lower Highs (red triangle down)
    if not lh_points.empty:
        fig.add_trace(
            go.Scatter(
                x=lh_points.index,
                y=lh_points['High'] * 1.01,
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=12, color='#ff4444'),
                text=['LH'] * len(lh_points),
                textposition='top center',
                textfont=dict(color='#ff4444', size=10),
                name='Lower Highs',
                showlegend=True
            ),
            row=row, col=col
        )
    
    # Higher Lows (green triangle up)
    if not hl_points.empty:
        fig.add_trace(
            go.Scatter(
                x=hl_points.index,
                y=hl_points['Low'] * 0.99,
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=12, color='#00ff00'),
                text=['HL'] * len(hl_points),
                textposition='bottom center',
                textfont=dict(color='#00ff00', size=10),
                name='Higher Lows',
                showlegend=True
            ),
            row=row, col=col
        )
    
    # Lower Lows (red triangle down)
    if not ll_points.empty:
        fig.add_trace(
            go.Scatter(
                x=ll_points.index,
                y=ll_points['Low'] * 0.99,
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=12, color='#ff4444'),
                text=['LL'] * len(ll_points),
                textposition='bottom center',
                textfont=dict(color='#ff4444', size=10),
                name='Lower Lows',
                showlegend=True
            ),
            row=row, col=col
        )
    
    return fig


def plot_candlestick_patterns(fig, data, row=1, col=1):
    """
    Mark detected candlestick patterns on chart.
    
    Args:
        fig: Plotly figure
        data (pd.DataFrame): Stock data with pattern columns
        row (int): Subplot row
        col (int): Subplot column
    
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    import plotly.graph_objects as go
    
    if 'Pattern' not in data.columns:
        return fig
    
    patterns = data[data['Pattern'] != '']
    
    # Pattern marker configurations
    pattern_config = {
        'PinBar': {'symbol': 'star', 'color': '#FFD700', 'size': 14},
        'Engulfing': {'symbol': 'diamond', 'color': '#FF69B4', 'size': 14},
        'InsideBar': {'symbol': 'square', 'color': '#87CEEB', 'size': 10},
        'ImpulseCandle': {'symbol': 'arrow-up', 'color': '#9370DB', 'size': 12},
        'Doji': {'symbol': 'cross', 'color': '#FFA500', 'size': 10}
    }
    
    for pattern_type, config in pattern_config.items():
        pattern_data = patterns[patterns['Pattern'] == pattern_type]
        
        if not pattern_data.empty:
            # Position markers above high for bearish, below low for bullish
            y_positions = []
            for idx, row_data in pattern_data.iterrows():
                if row_data.get('PatternDirection', '') == 'Bearish':
                    y_positions.append(row_data['High'] * 1.015)
                else:
                    y_positions.append(row_data['Low'] * 0.985)
            
            fig.add_trace(
                go.Scatter(
                    x=pattern_data.index,
                    y=y_positions,
                    mode='markers',
                    marker=dict(
                        symbol=config['symbol'],
                        size=config['size'],
                        color=config['color'],
                        line=dict(color='white', width=1)
                    ),
                    name=pattern_type,
                    hovertemplate=f"{pattern_type}<br>%{{x}}<extra></extra>",
                    showlegend=True
                ),
                row=row, col=col
            )
    
    return fig


def plot_trading_signals(fig, data, row=1, col=1):
    """
    Display trading signals on chart with entry, stop, and target levels.
    
    Args:
        fig: Plotly figure
        data (pd.DataFrame): Stock data with signal columns
        row (int): Subplot row
        col (int): Subplot column
    
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    import plotly.graph_objects as go
    
    if 'Signal' not in data.columns:
        return fig
    
    signals = data[data['Signal'] != '']
    
    buy_signals = signals[signals['Signal'] == 'BUY']
    sell_signals = signals[signals['Signal'] == 'SELL']
    
    # Plot BUY signals
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Entry'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=16,
                    color='#00FF00',
                    line=dict(color='white', width=2)
                ),
                name='BUY Signal',
                hovertemplate=(
                    "BUY<br>" +
                    "Entry: $%{y:.2f}<br>" +
                    "Stop: $%{customdata[0]:.2f}<br>" +
                    "Target: $%{customdata[1]:.2f}<br>" +
                    "R:R: %{customdata[2]:.1f}<extra></extra>"
                ),
                customdata=buy_signals[['StopLoss', 'TakeProfit', 'RR_Ratio']].values,
                showlegend=True
            ),
            row=row, col=col
        )
        
        # Add stop loss markers for BUY signals
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['StopLoss'],
                mode='markers',
                marker=dict(symbol='line-ew', size=12, color='red', line=dict(width=2)),
                name='Stop Loss (BUY)',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add take profit markers for BUY signals
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['TakeProfit'],
                mode='markers',
                marker=dict(symbol='line-ew', size=12, color='#00FF00', line=dict(width=2)),
                name='Take Profit (BUY)',
                showlegend=False
            ),
            row=row, col=col
        )
    
    # Plot SELL signals
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Entry'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=16,
                    color='#FF4444',
                    line=dict(color='white', width=2)
                ),
                name='SELL Signal',
                hovertemplate=(
                    "SELL<br>" +
                    "Entry: $%{y:.2f}<br>" +
                    "Stop: $%{customdata[0]:.2f}<br>" +
                    "Target: $%{customdata[1]:.2f}<br>" +
                    "R:R: %{customdata[2]:.1f}<extra></extra>"
                ),
                customdata=sell_signals[['StopLoss', 'TakeProfit', 'RR_Ratio']].values,
                showlegend=True
            ),
            row=row, col=col
        )
        
        # Add stop loss markers for SELL signals
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['StopLoss'],
                mode='markers',
                marker=dict(symbol='line-ew', size=12, color='red', line=dict(width=2)),
                name='Stop Loss (SELL)',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add take profit markers for SELL signals
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['TakeProfit'],
                mode='markers',
                marker=dict(symbol='line-ew', size=12, color='#FF4444', line=dict(width=2)),
                name='Take Profit (SELL)',
                showlegend=False
            ),
            row=row, col=col
        )
    
    return fig


def plot_price_action_dashboard(data, ticker, sr_data=None, show_patterns=True, show_structure=True, show_signals=True):
    """
    Create comprehensive price action analysis dashboard.
    
    Args:
        data (pd.DataFrame): Stock data with price action columns
        ticker (str): Stock ticker symbol
        sr_data (dict): Support/resistance data
        show_patterns (bool): Show candlestick patterns
        show_structure (bool): Show trend structure
        show_signals (bool): Show trading signals
    
    Returns:
        plotly.graph_objects.Figure: Complete price action dashboard
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Determine number of rows
    num_rows = 2  # Price and Volume minimum
    subplot_titles = ["Price Action Analysis", "Volume"]
    row_heights = [0.7, 0.15]
    
    # Add market phase indicator if available
    if 'MarketPhase' in data.columns:
        num_rows += 1
        subplot_titles.append("Market Phase")
        row_heights.append(0.15)
    
    # Normalize row heights
    row_heights = [h / sum(row_heights) for h in row_heights]
    
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f"{ticker}",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add EMA if present
    if 'EMA' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA'],
                name='EMA',
                line=dict(color='#FFD700', width=1.5, dash='dot')
            ),
            row=1, col=1
        )
    
    # Add S/R zones
    if sr_data:
        fig = plot_support_resistance_zones(fig, data, sr_data, row=1, col=1)
    
    # Add trend structure
    if show_structure and 'SwingType' in data.columns:
        fig = plot_trend_structure(fig, data, row=1, col=1)
    
    # Add candlestick patterns
    if show_patterns and 'Pattern' in data.columns:
        fig = plot_candlestick_patterns(fig, data, row=1, col=1)
    
    # Add trading signals
    if show_signals and 'Signal' in data.columns:
        fig = plot_trading_signals(fig, data, row=1, col=1)
    
    # Volume bars
    colors = ['#FF4444' if close < open else '#00FF00' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Market phase indicator
    if 'MarketPhase' in data.columns:
        phase_colors = {
            'Trending': '#00FF00',
            'Ranging': '#FFD700',
            'Consolidating': '#87CEEB',
            'Unknown': '#808080'
        }
        
        phase_values = []
        phase_colors_list = []
        for phase in data['MarketPhase']:
            if phase == 'Trending':
                phase_values.append(1)
            elif phase == 'Ranging':
                phase_values.append(0.5)
            elif phase == 'Consolidating':
                phase_values.append(0.25)
            else:
                phase_values.append(0)
            phase_colors_list.append(phase_colors.get(phase, '#808080'))
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=phase_values,
                marker_color=phase_colors_list,
                name='Market Phase',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Price Action Analysis",
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_signal_summary_table(data, ticker):
    """
    Create a summary table of trading signals.
    
    Args:
        data (pd.DataFrame): Data with signal columns
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: Signal summary table
    """
    if 'Signal' not in data.columns:
        return pd.DataFrame()
    
    signals = data[data['Signal'] != ''].copy()
    
    if signals.empty:
        return pd.DataFrame()
    
    summary_data = []
    for idx, row in signals.iterrows():
        summary_data.append({
            'Date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
            'Signal': row['Signal'],
            'Type': row.get('SignalType', 'N/A'),
            'Entry': f"${row['Entry']:.2f}" if pd.notna(row.get('Entry')) else 'N/A',
            'Stop': f"${row['StopLoss']:.2f}" if pd.notna(row.get('StopLoss')) else 'N/A',
            'Target': f"${row['TakeProfit']:.2f}" if pd.notna(row.get('TakeProfit')) else 'N/A',
            'R:R': f"{row['RR_Ratio']:.1f}" if pd.notna(row.get('RR_Ratio')) else 'N/A',
            'Strength': row.get('SignalStrength', 'N/A')
        })
    
    return pd.DataFrame(summary_data)
