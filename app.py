import streamlit as st
from modules.data_fetcher import fetch_stock_data, get_stock_info, validate_date_range
from modules.indicators import sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator, williams_r, atr, calculate_all_indicators
from modules.plotter import plot_stock_dashboard, plot_comparison_chart, create_summary_metrics_table, plot_price_action_dashboard, create_signal_summary_table
from modules.utils import export_data_to_csv, format_number, calculate_portfolio_metrics, get_popular_tickers, validate_tickers, create_indicator_config
from modules.price_action import analyze_price_action, detect_support_resistance, identify_trend_structure, detect_candlestick_patterns
from modules.strategies import generate_trading_signals, level_signal_rr_strategy, trend_pullback_strategy, breakout_retest_strategy, get_signal_summary
# Advanced modules
from modules.backtesting import BacktestEngine, BacktestConfig, walk_forward_analysis, format_metrics_table
from modules.monte_carlo import monte_carlo_simulation, format_monte_carlo_summary
from modules.risk_management import position_size_fixed_pct, format_position_size_table
from modules.trade_journal import TradeJournal, JournalEntry, format_journal_summary
from modules.mtf_analysis import analyze_mtf, format_mtf_summary
from modules.ml_patterns import analyze_patterns, format_patterns_table
from modules.ml_forecasting import forecast_price, format_forecast_table
from modules.alerts import AlertManager, AlertType, format_alerts_table
from modules.paper_trading import PaperTradingAccount, format_positions_table, format_portfolio_summary
from modules.sentiment import analyze_news_sentiment, format_sentiment_summary
from modules.scanner import scan_watchlist, ScanCriteria, quick_scan_universe, format_scan_results
# New modules
from modules.replay import MarketReplay, format_replay_summary
from modules.multi_asset import get_crypto_list, get_forex_list, get_commodity_list, fetch_multi_asset_data, compare_assets, format_intermarket_summary, analyze_intermarket_relationships, CRYPTO_SYMBOLS, FOREX_PAIRS, COMMODITY_SYMBOLS
from modules.plugins import get_plugin_manager, create_indicator_from_formula
from datetime import date, datetime, timedelta
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="📈 Stock Analysis Dashboard", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 Stock Price Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Configuration")

# Analysis mode selection
analysis_modes = ["Single Stock", "Portfolio Comparison", "Popular Picks", "Price Action Analysis", "Backtesting Lab", "Paper Trading", "Stock Scanner", "Market Replay", "Multi-Asset"]
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    analysis_modes,
    index=0 if "analysis_mode" not in st.session_state or st.session_state.analysis_mode not in analysis_modes else analysis_modes.index(st.session_state.analysis_mode)
)

# Update session state with current selection
st.session_state.analysis_mode = analysis_mode

if analysis_mode == "Single Stock":
    # Single stock analysis
    st.sidebar.subheader("Stock Selection")
    
    # Use session state ticker if available, otherwise default to AAPL
    default_ticker = st.session_state.get("ticker", "AAPL")
    ticker = st.sidebar.text_input("Enter Stock Ticker:", default_ticker).upper()
    
    # Update session state with current ticker
    st.session_state.ticker = ticker
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    
    # Quick date range buttons
    date_option = st.sidebar.selectbox(
        "Quick Select:",
        ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"]
    )
    
    if date_option == "Custom":
        start = st.sidebar.date_input("Start Date", date(2023, 1, 1))
        end = st.sidebar.date_input("End Date", date.today())
    else:
        end = date.today()
        if date_option == "1 Month":
            start = end - timedelta(days=30)
        elif date_option == "3 Months":
            start = end - timedelta(days=90)
        elif date_option == "6 Months":
            start = end - timedelta(days=180)
        elif date_option == "1 Year":
            start = end - timedelta(days=365)
        elif date_option == "2 Years":
            start = end - timedelta(days=730)
        elif date_option == "5 Years":
            start = end - timedelta(days=1825)
    
    # Technical Indicators
    st.sidebar.subheader("📊 Technical Indicators")
    
    indicators = []
    # Moving Averages
    with st.sidebar.expander("Moving Averages", expanded=True):
        if st.checkbox("20-day SMA"): indicators.append("SMA20")
        if st.checkbox("50-day SMA"): indicators.append("SMA50")
        if st.checkbox("200-day SMA"): indicators.append("SMA200")
        if st.checkbox("20-day EMA"): indicators.append("EMA20")
        if st.checkbox("Bollinger Bands"): indicators.append("Bollinger")
    
    # Momentum Indicators
    with st.sidebar.expander("Momentum Indicators"):
        if st.checkbox("RSI (14)"): indicators.append("RSI")
        if st.checkbox("MACD"): indicators.append("MACD")
        if st.checkbox("Stochastic"): indicators.append("Stochastic")
        if st.checkbox("Williams %R"): indicators.append("Williams_R")
        if st.checkbox("ATR"): indicators.append("ATR")
    
    # Analysis button
    if st.sidebar.button("🚀 Analyze Stock", type="primary"):
        if validate_date_range(start, end):
            # Fetch stock data
            data = fetch_stock_data(ticker, start, end)
            
            if not data.empty:
                # Get stock info
                stock_info = get_stock_info(ticker)
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart Analysis", "📊 Summary", "💾 Data Export", "ℹ️ Stock Info"])
                
                with tab1:
                    # Calculate indicators
                    if "SMA20" in indicators: data = sma(data, 20)
                    if "SMA50" in indicators: data = sma(data, 50)
                    if "SMA200" in indicators: data = sma(data, 200)
                    if "EMA20" in indicators: data = ema(data, 20)
                    if "RSI" in indicators: data = rsi(data)
                    if "MACD" in indicators: data = macd(data)
                    if "Bollinger" in indicators: data = bollinger_bands(data)
                    if "Stochastic" in indicators: data = stochastic_oscillator(data)
                    if "Williams_R" in indicators: data = williams_r(data)
                    if "ATR" in indicators: data = atr(data)
                    
                    # Plot dashboard
                    fig = plot_stock_dashboard(data, ticker, indicators)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Summary metrics
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📈 Key Metrics")
                        metrics_df = create_summary_metrics_table(data, ticker)
                        if not metrics_df.empty:
                            st.dataframe(metrics_df, hide_index=True)
                    
                    with col2:
                        st.subheader("🎯 Technical Signals")
                        if not data.empty:
                            current_price = data['Close'].iloc[-1]
                            
                            # RSI Signal
                            if 'RSI' in data.columns:
                                current_rsi = data['RSI'].iloc[-1]
                                if current_rsi > 70:
                                    st.error(f"🔴 RSI: {current_rsi:.1f} - Overbought")
                                elif current_rsi < 30:
                                    st.success(f"🟢 RSI: {current_rsi:.1f} - Oversold")
                                else:
                                    st.info(f"🔵 RSI: {current_rsi:.1f} - Neutral")
                            
                            # MACD Signal
                            if 'MACD' in data.columns and 'Signal' in data.columns:
                                macd_current = data['MACD'].iloc[-1]
                                signal_current = data['Signal'].iloc[-1]
                                if macd_current > signal_current:
                                    st.success("🟢 MACD: Bullish")
                                else:
                                    st.error("🔴 MACD: Bearish")
                            
                            # Moving Average Signal
                            if 'SMA50' in data.columns:
                                sma50_current = data['SMA50'].iloc[-1]
                                if current_price > sma50_current:
                                    st.success("🟢 Price above 50-day SMA")
                                else:
                                    st.error("🔴 Price below 50-day SMA")
                
                with tab3:
                    st.subheader("💾 Export Data")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📥 Download Raw Data"):
                            csv_link = export_data_to_csv(data, f"{ticker}_raw_data")
                            if csv_link:
                                st.markdown(csv_link, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("📊 Download with Indicators"):
                            csv_link = export_data_to_csv(data, f"{ticker}_with_indicators")
                            if csv_link:
                                st.markdown(csv_link, unsafe_allow_html=True)
                    
                    # Display data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(data.tail(10))
                
                with tab4:
                    st.subheader("ℹ️ Company Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Company Name:** {stock_info['name']}")
                        st.write(f"**Sector:** {stock_info['sector']}")
                        st.write(f"**Industry:** {stock_info['industry']}")
                    
                    with col2:
                        st.write(f"**Market Cap:** {format_number(stock_info['market_cap'], 'market_cap')}")
                        st.write(f"**P/E Ratio:** {stock_info['pe_ratio']}")

elif analysis_mode == "Portfolio Comparison":
    # Portfolio comparison mode
    st.sidebar.subheader("Portfolio Setup")
    
    # Multiple ticker input
    tickers_input = st.sidebar.text_area(
        "Enter tickers (one per line):",
        value="AAPL\nMSFT\nGOOGL\nAMZN",
        height=100
    )
    
    tickers = [ticker.strip().upper() for ticker in tickers_input.split('\n') if ticker.strip()]
    tickers = validate_tickers(tickers)
    
    if len(tickers) > 10:
        st.sidebar.warning("⚠️ Maximum 10 stocks allowed for comparison")
        tickers = tickers[:10]
    
    # Date range
    end_date = date.today()
    start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
    
    if st.sidebar.button("🔍 Compare Stocks", type="primary"):
        if len(tickers) >= 2:
            data_dict = {}
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                data = fetch_stock_data(ticker, start_date, end_date)
                if not data.empty:
                    data_dict[ticker] = data
                progress_bar.progress((i + 1) / len(tickers))
            
            progress_bar.empty()
            
            if len(data_dict) >= 2:
                # Create comparison chart
                comparison_fig = plot_comparison_chart(data_dict, list(data_dict.keys()))
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Portfolio metrics
                st.subheader("📊 Portfolio Analysis")
                portfolio_metrics = calculate_portfolio_metrics(data_dict)
                
                if portfolio_metrics:
                    cols = st.columns(len(portfolio_metrics))
                    for i, (metric, value) in enumerate(portfolio_metrics.items()):
                        cols[i].metric(metric, value)
                
                # Individual stock performance table
                st.subheader("📈 Individual Performance")
                performance_data = []
                
                for ticker, data in data_dict.items():
                    if not data.empty:
                        start_price = data['Close'].iloc[0].item()
                        end_price = data['Close'].iloc[-1].item()
                        total_return = ((end_price - start_price) / start_price) * 100
                        
                        performance_data.append({
                            'Ticker': ticker,
                            'Start Price': f"${start_price:.2f}",
                            'Current Price': f"${end_price:.2f}",
                            'Total Return': f"{total_return:.2f}%"
                        })
                
                if performance_data:
                    st.dataframe(pd.DataFrame(performance_data), hide_index=True)
            else:
                st.error("⚠️ Could not fetch data for enough stocks to compare")
        else:
            st.error("⚠️ Please enter at least 2 ticker symbols")

elif analysis_mode == "Popular Picks":
    # Popular stocks mode
    st.sidebar.subheader("Popular Categories")
    
    popular_tickers = get_popular_tickers()
    selected_category = st.sidebar.selectbox("Choose Category:", list(popular_tickers.keys()))
    
    category_tickers = popular_tickers[selected_category]
    
    # Display popular tickers
    st.subheader(f"📈 {selected_category} Stocks")
    
    cols = st.columns(3)
    for i, ticker in enumerate(category_tickers):
        with cols[i % 3]:
            if st.button(f"📊 Analyze {ticker}", key=f"popular_{ticker}"):
                # Switch to single stock analysis mode and set ticker
                st.session_state.analysis_mode = "Single Stock"
                st.session_state.ticker = ticker
                st.rerun()
    
    # Quick comparison for category
    if st.sidebar.button(f"🔍 Compare All {selected_category}"):
        end_date = date.today()
        start_date = end_date - timedelta(days=365)  # 1 year comparison
        
        data_dict = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(category_tickers):
            data = fetch_stock_data(ticker, start_date, end_date)
            if not data.empty:
                data_dict[ticker] = data
            progress_bar.progress((i + 1) / len(category_tickers))
        
        progress_bar.empty()
        
        if data_dict:
            comparison_fig = plot_comparison_chart(data_dict, list(data_dict.keys()))
            st.plotly_chart(comparison_fig, use_container_width=True)

elif analysis_mode == "Price Action Analysis":
    # Price Action Analysis mode
    st.sidebar.subheader("🎯 Price Action Setup")
    
    # Stock selection
    default_ticker = st.session_state.get("ticker", "AAPL")
    ticker = st.sidebar.text_input("Enter Stock Ticker:", default_ticker).upper()
    st.session_state.ticker = ticker
    
    # Date range
    st.sidebar.subheader("Date Range")
    date_option = st.sidebar.selectbox(
        "Quick Select:",
        ["6 Months", "1 Year", "2 Years", "Custom"],
        index=1
    )
    
    if date_option == "Custom":
        start = st.sidebar.date_input("Start Date", date(2023, 1, 1))
        end = st.sidebar.date_input("End Date", date.today())
    else:
        end = date.today()
        if date_option == "6 Months":
            start = end - timedelta(days=180)
        elif date_option == "1 Year":
            start = end - timedelta(days=365)
        elif date_option == "2 Years":
            start = end - timedelta(days=730)
    
    # Strategy selection
    st.sidebar.subheader("📊 Strategy Selection")
    selected_strategy = st.sidebar.selectbox(
        "Choose Strategy:",
        ["Level + Signal + RR", "Trend Pullback", "Breakout & Retest", "All Strategies"]
    )
    
    # Strategy parameters
    with st.sidebar.expander("⚙️ Strategy Parameters", expanded=False):
        min_rr = st.slider("Minimum R:R Ratio", 1.5, 4.0, 2.0, 0.5)
        swing_lookback = st.slider("Swing Lookback", 3, 10, 5)
        sr_threshold = st.slider("S/R Threshold (%)", 1.0, 5.0, 2.0, 0.5) / 100
        ema_period = st.slider("EMA Period", 10, 50, 20)
    
    # Visualization options
    with st.sidebar.expander("👁️ Display Options", expanded=True):
        show_sr_zones = st.checkbox("Show S/R Zones", value=True)
        show_trend_structure = st.checkbox("Show Trend Structure", value=True)
        show_patterns = st.checkbox("Show Candlestick Patterns", value=True)
        show_signals = st.checkbox("Show Trading Signals", value=True)
    
    # Analyze button
    if st.sidebar.button("🎯 Analyze Price Action", type="primary"):
        if validate_date_range(start, end):
            # Fetch stock data
            data = fetch_stock_data(ticker, start, end)
            
            if not data.empty:
                # Get stock info
                stock_info = get_stock_info(ticker)
                
                # Create tabs for price action analysis
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Price Action Chart", "🎯 Trading Signals", 
                    "📊 Pattern Analysis", "ℹ️ Summary"
                ])
                
                with tab1:
                    st.subheader(f"🎯 {ticker} - Price Action Analysis")
                    
                    # Perform full price action analysis
                    analyzed_data, sr_data, pa_summary = analyze_price_action(
                        data, 
                        swing_lookback=swing_lookback,
                        sr_lookback=20,
                        sr_threshold=sr_threshold
                    )
                    
                    # Apply selected strategy
                    strategy_map = {
                        "Level + Signal + RR": "level_signal_rr",
                        "Trend Pullback": "trend_pullback",
                        "Breakout & Retest": "breakout_retest"
                    }
                    
                    if selected_strategy == "All Strategies":
                        # Apply all strategies
                        for strat_name, strat_key in strategy_map.items():
                            params = {
                                'min_rr': min_rr,
                                'sr_lookback': 20,
                                'sr_threshold': sr_threshold,
                                'ema_period': ema_period,
                                'swing_lookback': swing_lookback,
                                'consolidation_bars': 10,
                                'confirmation_required': True
                            }
                            analyzed_data, _, _ = generate_trading_signals(
                                analyzed_data, strat_key, params
                            )
                    else:
                        strat_key = strategy_map[selected_strategy]
                        params = {
                            'min_rr': min_rr,
                            'sr_lookback': 20,
                            'sr_threshold': sr_threshold,
                            'ema_period': ema_period,
                            'swing_lookback': swing_lookback,
                            'consolidation_bars': 10,
                            'confirmation_required': True
                        }
                        analyzed_data, sr_data_strat, signal_summary = generate_trading_signals(
                            analyzed_data, strat_key, params
                        )
                        if sr_data_strat:
                            sr_data = sr_data_strat
                    
                    # Plot price action dashboard
                    fig = plot_price_action_dashboard(
                        analyzed_data, 
                        ticker, 
                        sr_data=sr_data if show_sr_zones else None,
                        show_patterns=show_patterns,
                        show_structure=show_trend_structure,
                        show_signals=show_signals
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Quick stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        trend = pa_summary.get('trend', 'N/A')
                        trend_color = "🟢" if trend == "Uptrend" else "🔴" if trend == "Downtrend" else "🟡"
                        st.metric("Current Trend", f"{trend_color} {trend}")
                    with col2:
                        phase = pa_summary.get('phase', 'N/A')
                        st.metric("Market Phase", phase)
                    with col3:
                        support_count = len(sr_data.get('support_levels', []))
                        st.metric("Support Levels", support_count)
                    with col4:
                        resistance_count = len(sr_data.get('resistance_levels', []))
                        st.metric("Resistance Levels", resistance_count)
                
                with tab2:
                    st.subheader("🎯 Trading Signals")
                    
                    # Signal summary
                    signal_summary = get_signal_summary(analyzed_data)
                    
                    if signal_summary.get('total_signals', 0) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Signals", signal_summary['total_signals'])
                        with col2:
                            st.metric("BUY Signals", signal_summary['buy_signals'])
                        with col3:
                            st.metric("SELL Signals", signal_summary['sell_signals'])
                        with col4:
                            avg_rr = signal_summary.get('avg_rr', 0)
                            st.metric("Avg R:R", f"{avg_rr:.1f}" if avg_rr else "N/A")
                        
                        # Signal table
                        st.subheader("📋 Signal Details")
                        signal_table = create_signal_summary_table(analyzed_data, ticker)
                        if not signal_table.empty:
                            st.dataframe(signal_table, hide_index=True, use_container_width=True)
                        
                        # Recent signals highlight
                        st.subheader("🔔 Recent Signals")
                        recent = signal_summary.get('recent_signals', [])
                        for sig in recent[-3:]:
                            signal_type = sig.get('Signal', '')
                            signal_style = sig.get('SignalType', '')
                            entry = sig.get('Entry', 0)
                            stop = sig.get('StopLoss', 0)
                            target = sig.get('TakeProfit', 0)
                            rr = sig.get('RR_Ratio', 0)
                            
                            if signal_type == 'BUY':
                                st.success(f"🟢 **BUY** ({signal_style}) - Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f} | R:R: {rr:.1f}")
                            else:
                                st.error(f"🔴 **SELL** ({signal_style}) - Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f} | R:R: {rr:.1f}")
                    else:
                        st.info("📊 No trading signals detected with current parameters. Try adjusting the R:R ratio or lookback periods.")
                
                with tab3:
                    st.subheader("📊 Candlestick Pattern Analysis")
                    
                    if 'Pattern' in analyzed_data.columns:
                        patterns = analyzed_data[analyzed_data['Pattern'] != '']
                        
                        if not patterns.empty:
                            # Pattern summary
                            pattern_counts = patterns['Pattern'].value_counts()
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.write("**Pattern Counts:**")
                                for pattern, count in pattern_counts.items():
                                    st.write(f"- {pattern}: {count}")
                            
                            with col2:
                                # Recent patterns table
                                recent_patterns = patterns.tail(10)[['Pattern', 'PatternDirection', 'Close']].copy()
                                recent_patterns['Close'] = recent_patterns['Close'].apply(lambda x: f"${x:.2f}")
                                recent_patterns.index = recent_patterns.index.strftime('%Y-%m-%d')
                                st.write("**Recent Patterns:**")
                                st.dataframe(recent_patterns, use_container_width=True)
                            
                            # Pattern legend
                            st.markdown("""
                            ---
                            **Pattern Legend:**
                            - ⭐ **Pin Bar**: Long wick rejection candle (reversal signal)
                            - 💎 **Engulfing**: Current candle engulfs previous (strong reversal)
                            - ⬜ **Inside Bar**: Range within previous bar (breakout pending)
                            - 🔮 **Impulse Candle**: Large body with strong momentum
                            - ➕ **Doji**: Very small body (indecision)
                            """)
                        else:
                            st.info("No candlestick patterns detected in the selected period.")
                    
                    # Support/Resistance levels
                    st.subheader("📍 Key Levels")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🟢 Support Levels:**")
                        for i, level in enumerate(sr_data.get('support_levels', [])[:5]):
                            st.write(f"{i+1}. ${level['level']:.2f} (Strength: {level['strength']})")
                    
                    with col2:
                        st.write("**🔴 Resistance Levels:**")
                        for i, level in enumerate(sr_data.get('resistance_levels', [])[:5]):
                            st.write(f"{i+1}. ${level['level']:.2f} (Strength: {level['strength']})")
                
                with tab4:
                    st.subheader("ℹ️ Analysis Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📈 Stock Information:**")
                        st.write(f"- **Company:** {stock_info['name']}")
                        st.write(f"- **Sector:** {stock_info['sector']}")
                        st.write(f"- **Industry:** {stock_info['industry']}")
                        st.write(f"- **Market Cap:** {format_number(stock_info['market_cap'], 'market_cap')}")
                        
                        current_price = data['Close'].iloc[-1]
                        st.write(f"- **Current Price:** ${current_price:.2f}")
                    
                    with col2:
                        st.write("**🎯 Strategy Applied:**")
                        st.write(f"- **Strategy:** {selected_strategy}")
                        st.write(f"- **Min R:R:** {min_rr}")
                        st.write(f"- **Swing Lookback:** {swing_lookback}")
                        st.write(f"- **S/R Threshold:** {sr_threshold*100:.1f}%")
                        st.write(f"- **EMA Period:** {ema_period}")
                    
                    # Export option
                    st.subheader("💾 Export Data")
                    if st.button("📥 Download Analysis Data"):
                        csv_link = export_data_to_csv(analyzed_data, f"{ticker}_price_action")
                        if csv_link:
                            st.markdown(csv_link, unsafe_allow_html=True)
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Disclaimer:** Price action analysis and trading signals are for educational purposes only. 
                    Past performance does not guarantee future results. Always use proper risk management 
                    and consult with a financial advisor before making trading decisions.
                    """)
            else:
                st.error(f"⚠️ Could not fetch data for {ticker}. Please check the ticker symbol.")

elif analysis_mode == "Backtesting Lab":
    # Backtesting mode
    st.sidebar.subheader("🔬 Backtesting Setup")
    
    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker:", "AAPL").upper()
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start = st.date_input("Start", date.today() - timedelta(days=730))
    with col2:
        end = st.date_input("End", date.today())
    
    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Strategy:",
        ["Level + Signal + RR", "Trend Pullback", "Breakout & Retest"]
    )
    
    # Backtest config
    with st.sidebar.expander("⚙️ Backtest Settings"):
        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
        slippage = st.slider("Slippage (%)", 0.0, 0.5, 0.1, 0.05)
    
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        data = fetch_stock_data(ticker, start, end)
        
        if not data.empty:
            st.subheader(f"🔬 {ticker} Backtest Results")
            
            # Generate signals
            strategy_map = {
                "Level + Signal + RR": "level_signal_rr",
                "Trend Pullback": "trend_pullback",
                "Breakout & Retest": "breakout_retest"
            }
            strat_key = strategy_map[strategy]
            
            analyzed_data, sr_data, signal_summary = generate_trading_signals(
                data.copy(), strat_key, {'min_rr': 2.0}
            )
            
            # Run backtest
            config = BacktestConfig(
                initial_capital=initial_capital,
                slippage_pct=slippage / 100,
                risk_per_trade_pct=risk_per_trade / 100
            )
            
            engine = BacktestEngine(config)
            results = engine.run_backtest(analyzed_data, analyzed_data)
            
            # Display results
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Equity Curve", "📋 Trades", "🎲 Monte Carlo"])
            
            with tab1:
                metrics = results['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.1f}%")
                with col2:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
                with col3:
                    st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                with col4:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", metrics.get('total_trades', 0))
                with col2:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
                with col3:
                    st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
                with col4:
                    st.metric("Final Capital", f"${metrics.get('final_capital', 0):,.0f}")
                
                st.dataframe(format_metrics_table(metrics), hide_index=True)
            
            with tab2:
                equity_df = results['equity_curve']
                if not equity_df.empty:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=equity_df['date'], y=equity_df['equity'], 
                                            mode='lines', name='Equity Curve',
                                            line=dict(color='#00FF00', width=2)))
                    fig.update_layout(title=f'{ticker} Equity Curve', 
                                     xaxis_title='Date', yaxis_title='Equity ($)',
                                     template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if results['trades']:
                    trades_data = [{
                        'Entry': t.entry_date.strftime('%Y-%m-%d') if hasattr(t.entry_date, 'strftime') else str(t.entry_date)[:10],
                        'Exit': t.exit_date.strftime('%Y-%m-%d') if hasattr(t.exit_date, 'strftime') else str(t.exit_date)[:10],
                        'Side': t.side.value,
                        'Entry $': f"${t.entry_price:.2f}",
                        'Exit $': f"${t.exit_price:.2f}",
                        'P&L': f"${t.pnl:.2f}",
                        'R': f"{t.r_multiple:.1f}R"
                    } for t in results['trades']]
                    st.dataframe(pd.DataFrame(trades_data), hide_index=True)
                else:
                    st.info("No trades executed")
            
            with tab4:
                if results['trades']:
                    trade_pnls = [t.pnl for t in results['trades']]
                    mc_result = monte_carlo_simulation(trade_pnls, initial_capital, n_simulations=500)
                    st.dataframe(format_monte_carlo_summary(mc_result, initial_capital), hide_index=True)
                    st.metric("Risk of Ruin", f"{mc_result.risk_of_ruin*100:.1f}%")
        else:
            st.error("Could not fetch data")

elif analysis_mode == "Paper Trading":
    # Paper trading mode
    st.sidebar.subheader("📝 Paper Trading")
    
    # Initialize account
    if 'paper_account' not in st.session_state:
        st.session_state.paper_account = PaperTradingAccount(initial_balance=100000)
    
    account = st.session_state.paper_account
    
    # Trade form
    st.sidebar.subheader("Place Order")
    order_symbol = st.sidebar.text_input("Symbol:", "AAPL").upper()
    order_side = st.sidebar.selectbox("Side:", ["BUY", "SELL"])
    order_qty = st.sidebar.number_input("Quantity:", 1, 10000, 10)
    
    # Get current price
    current_price = 0
    if order_symbol:
        try:
            price_data = fetch_stock_data(order_symbol, date.today() - timedelta(days=5), date.today())
            if not price_data.empty:
                current_price = price_data['Close'].iloc[-1]
                st.sidebar.write(f"Current Price: ${current_price:.2f}")
        except:
            pass
    
    if st.sidebar.button("📤 Place Order"):
        if current_price > 0:
            try:
                order = account.place_order(
                    order_symbol, order_side, order_qty, current_price
                )
                st.sidebar.success(f"✅ {order_side} {order_qty} {order_symbol} @ ${current_price:.2f}")
            except ValueError as e:
                st.sidebar.error(str(e))
    
    if st.sidebar.button("🔄 Reset Account"):
        account.reset_account()
        st.sidebar.success("Account reset!")
    
    # Main display
    st.subheader("📝 Paper Trading Account")
    
    # Portfolio summary
    summary = account.get_portfolio_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cash", f"${summary['cash_balance']:,.2f}")
    with col2:
        st.metric("Positions Value", f"${summary['positions_value']:,.2f}")
    with col3:
        st.metric("Total Value", f"${summary['total_value']:,.2f}")
    with col4:
        return_color = "normal" if summary['total_return'] >= 0 else "inverse"
        st.metric("Return", f"{summary['total_return_pct']:.2f}%", delta_color=return_color)
    
    # Positions
    st.subheader("📊 Open Positions")
    positions = account.get_all_positions()
    if positions:
        st.dataframe(format_positions_table(positions), hide_index=True)
    else:
        st.info("No open positions")
    
    # Trade history
    st.subheader("📜 Trade History")
    trades = account.get_trade_history(20)
    if trades:
        trades_data = [{
            'Symbol': t.symbol,
            'Side': t.side,
            'Qty': t.quantity,
            'Entry': f"${t.entry_price:.2f}",
            'Exit': f"${t.exit_price:.2f}",
            'P&L': f"${t.pnl:.2f}",
            'Date': t.exit_date[:10] if t.exit_date else 'N/A'
        } for t in trades]
        st.dataframe(pd.DataFrame(trades_data), hide_index=True)
    else:
        st.info("No trade history")

elif analysis_mode == "Stock Scanner":
    # Scanner mode
    st.sidebar.subheader("🔍 Scanner Setup")
    
    universe = st.sidebar.selectbox(
        "Universe:",
        ["dow30", "tech", "finance", "healthcare", "energy"]
    )
    
    # Filters
    with st.sidebar.expander("🎯 Filters"):
        trend_filter = st.selectbox("Trend:", [None, "Uptrend", "Downtrend"])
        at_sr = st.checkbox("At S/R Level")
        has_signal = st.checkbox("Has Signal")
        min_score = st.slider("Min Score", 0, 100, 30)
    
    if st.sidebar.button("🔍 Scan Now", type="primary"):
        st.subheader(f"🔍 Scanning {universe.upper()} Stocks...")
        
        symbols = quick_scan_universe(universe)
        
        criteria = ScanCriteria(
            trend_filter=trend_filter,
            at_sr_level=at_sr,
            has_signal=has_signal,
            min_score=min_score
        )
        
        # Fetch and scan
        results = []
        progress = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            try:
                data = fetch_stock_data(symbol, date.today() - timedelta(days=365), date.today())
                if not data.empty:
                    from modules.scanner import scan_single_stock
                    result = scan_single_stock(symbol, data, criteria)
                    if result:
                        results.append(result)
            except:
                pass
            progress.progress((i + 1) / len(symbols))
        
        progress.empty()
        
        if results:
            st.success(f"Found {len(results)} opportunities!")
            st.dataframe(format_scan_results(results), hide_index=True)
            
            # Top picks
            st.subheader("🏆 Top 5 Opportunities")
            for i, r in enumerate(results[:5]):
                emoji = "🟢" if r.trend == "Uptrend" else "🔴" if r.trend == "Downtrend" else "🟡"
                st.write(f"{i+1}. **{r.symbol}** - {emoji} {r.trend} | Score: {r.score:.0f} | ${r.current_price:.2f}")
        else:
            st.info("No stocks match the criteria")

elif analysis_mode == "Market Replay":
    # Market Replay mode
    st.sidebar.subheader("🎬 Market Replay")
    
    # Stock selection
    replay_ticker = st.sidebar.text_input("Symbol:", "AAPL").upper()
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        replay_start = st.date_input("Start", date.today() - timedelta(days=365), key="replay_start")
    with col2:
        replay_end = st.date_input("End", date.today(), key="replay_end")
    
    # Initialize replay session
    if 'market_replay' not in st.session_state:
        st.session_state.market_replay = None
    if 'replay_symbol' not in st.session_state:
        st.session_state.replay_symbol = ""
    
    if st.sidebar.button("🎬 Load Data", type="primary"):
        data = fetch_stock_data(replay_ticker, replay_start, replay_end)
        if not data.empty:
            st.session_state.market_replay = MarketReplay(data, replay_ticker)
            st.session_state.replay_symbol = replay_ticker
            st.sidebar.success(f"Loaded {len(data)} bars")
    
    replay = st.session_state.market_replay
    
    if replay:
        st.subheader(f"🎬 Market Replay: {st.session_state.replay_symbol}")
        
        # Controls
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("⏮️ Reset"):
                replay.reset()
        with col2:
            if st.button("◀️ Back"):
                replay.step_backward()
        with col3:
            if st.button("▶️ Forward"):
                replay.step_forward()
        with col4:
            if st.button("⏩ +10 Bars"):
                for _ in range(10):
                    if not replay.step_forward():
                        break
        with col5:
            if st.button("⏭️ End"):
                replay.jump_to(replay.total_bars - 1)
        
        # Progress slider
        new_bar = st.slider("Bar Position", 0, replay.total_bars - 1, replay.current_bar)
        if new_bar != replay.current_bar:
            replay.jump_to(new_bar)
        
        # Trading controls
        st.subheader("💹 Trading Controls")
        col1, col2, col3 = st.columns(3)
        with col1:
            buy_qty = st.number_input("Quantity", 1, 1000, 10, key="buy_qty")
        with col2:
            if st.button("📈 BUY", type="primary"):
                if replay.buy(buy_qty, "Manual buy"):
                    st.success(f"Bought {buy_qty} @ ${replay.get_current_price():.2f}")
        with col3:
            if st.button("📉 SELL"):
                if replay.sell(buy_qty, "Manual sell"):
                    st.success(f"Sold {buy_qty} @ ${replay.get_current_price():.2f}")
        
        # Display summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chart
            visible_data = replay.get_visible_data()
            if not visible_data.empty:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Candlestick(
                    x=visible_data.index,
                    open=visible_data['Open'],
                    high=visible_data['High'],
                    low=visible_data['Low'],
                    close=visible_data['Close']
                )])
                fig.update_layout(title=f'{st.session_state.replay_symbol} - Bar {replay.current_bar + 1}/{replay.total_bars}',
                                 xaxis_rangeslider_visible=False, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            summary = replay.get_session_summary()
            st.dataframe(format_replay_summary(summary), hide_index=True)
        
        # Decisions
        st.subheader("📋 Trading Decisions")
        decisions_df = replay.get_decisions_df()
        if not decisions_df.empty:
            st.dataframe(decisions_df, hide_index=True)
        else:
            st.info("No decisions made yet")
    else:
        st.info("👆 Load data from sidebar to start market replay")

elif analysis_mode == "Multi-Asset":
    # Multi-Asset mode
    st.sidebar.subheader("🌐 Multi-Asset Analysis")
    
    asset_class = st.sidebar.selectbox(
        "Asset Class:",
        ["Crypto", "Forex", "Commodities", "Compare All"]
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ma_start = st.date_input("Start", date.today() - timedelta(days=180), key="ma_start")
    with col2:
        ma_end = st.date_input("End", date.today(), key="ma_end")
    
    if asset_class == "Crypto":
        st.subheader("🪙 Cryptocurrency Analysis")
        
        selected_crypto = st.multiselect(
            "Select Cryptocurrencies:",
            list(CRYPTO_SYMBOLS.keys()),
            default=['BTC-USD', 'ETH-USD']
        )
        
        if selected_crypto and st.button("📊 Analyze", type="primary"):
            data_dict = fetch_multi_asset_data(selected_crypto, str(ma_start), str(ma_end))
            
            if data_dict:
                # Comparison chart
                comparison = compare_assets(selected_crypto, str(ma_start), str(ma_end))
                if not comparison.empty:
                    import plotly.express as px
                    fig = px.line(comparison, title="Crypto Performance (Normalized to 100)")
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Individual metrics
                for symbol, data in data_dict.items():
                    if not data.empty:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{symbol}", f"${data['Close'].iloc[-1]:,.2f}")
                        with col2:
                            change = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                            st.metric("Period Return", f"{change:.1f}%")
                        with col3:
                            volatility = data['Close'].pct_change().std() * 100
                            st.metric("Daily Vol", f"{volatility:.2f}%")
    
    elif asset_class == "Forex":
        st.subheader("💱 Forex Analysis")
        
        selected_pairs = st.multiselect(
            "Select Currency Pairs:",
            list(FOREX_PAIRS.keys()),
            default=['EURUSD=X', 'GBPUSD=X']
        )
        
        if selected_pairs and st.button("📊 Analyze", type="primary"):
            comparison = compare_assets(selected_pairs, str(ma_start), str(ma_end))
            if not comparison.empty:
                import plotly.express as px
                fig = px.line(comparison, title="Forex Performance (Normalized)")
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    elif asset_class == "Commodities":
        st.subheader("🛢️ Commodities Analysis")
        
        selected_commodities = st.multiselect(
            "Select Commodities:",
            list(COMMODITY_SYMBOLS.keys()),
            default=['GC=F', 'CL=F']
        )
        
        if selected_commodities and st.button("📊 Analyze", type="primary"):
            comparison = compare_assets(selected_commodities, str(ma_start), str(ma_end))
            if not comparison.empty:
                import plotly.express as px
                fig = px.line(comparison, title="Commodities Performance (Normalized)")
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    elif asset_class == "Compare All":
        st.subheader("🌐 Cross-Asset Comparison")
        
        st.info("Compare major assets across different classes")
        
        compare_symbols = ['SPY', 'BTC-USD', 'GC=F', 'EURUSD=X']
        
        if st.button("📊 Run Comparison", type="primary"):
            data_dict = fetch_multi_asset_data(compare_symbols, str(ma_start), str(ma_end))
            
            if data_dict:
                # Performance comparison
                comparison = compare_assets(compare_symbols, str(ma_start), str(ma_end))
                if not comparison.empty:
                    import plotly.express as px
                    fig = px.line(comparison, title="Cross-Asset Performance")
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Intermarket analysis  
                spy_data = data_dict.get('SPY', pd.DataFrame())
                gold_data = data_dict.get('GC=F', pd.DataFrame())
                
                if not spy_data.empty:
                    intermarket = analyze_intermarket_relationships(spy_data, gold_data=gold_data)
                    st.subheader("📈 Intermarket Analysis")
                    st.dataframe(format_intermarket_summary(intermarket), hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📈 Stock Analysis Dashboard | Built with Streamlit, yfinance, and Plotly<br>
        ⚠️ For educational purposes only. Not financial advice.
    </div>
    """, 
    unsafe_allow_html=True
)
