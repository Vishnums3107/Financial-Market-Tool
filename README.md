# 📈 Stock Price Analysis Dashboard

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.50.0+-red.svg)
![Plotly](https://img.shields.io/badge/plotly-v6.3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An interactive web-based dashboard for analyzing historical stock prices with advanced technical indicators using Python, Pandas, yfinance, Plotly, and Streamlit. Perfect for demonstrating data analysis and web development skills in a resume-ready project.

## 🌟 Features

### Core Functionality
- **Real-time Stock Data**: Fetch historical stock data using yfinance API
- **Interactive Charts**: Beautiful, responsive charts with Plotly
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator, Williams %R, ATR
- **Multi-timeframe Analysis**: 1 month to 5 years of historical data
- **Portfolio Comparison**: Compare multiple stocks side-by-side
- **Popular Stock Categories**: Pre-defined lists of popular stocks by sector

### Advanced Features
- **Smart Date Selection**: Quick date range buttons and custom date picker
- **Technical Signals**: Automated buy/sell signals based on indicators
- **Data Export**: Export analysis data to CSV format
- **Stock Information**: Company details, market cap, P/E ratio, sector information
- **Portfolio Metrics**: Calculate portfolio-level performance metrics
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Technical Highlights
- **Modular Architecture**: Clean, maintainable code structure
- **Error Handling**: Robust error handling and user feedback
- **Caching**: Efficient data caching for improved performance
- **Unit Testing**: Comprehensive test suite with 100% pass rate
- **Documentation**: Well-documented code and deployment guides

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-dashboard.git
   cd stock-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   ```
   http://localhost:8501
   ```

## 📁 Project Structure

```
stock-dashboard/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── DEPLOYMENT.md         # Deployment guide
│
├── modules/              # Core modules
│   ├── data_fetcher.py   # Stock data acquisition
│   ├── indicators.py     # Technical indicators
│   ├── plotter.py        # Visualization functions
│   └── utils.py          # Utility functions
│
├── tests/                # Unit tests
│   ├── test_stock_dashboard.py
│   └── README.md
│
└── assets/               # Static assets
    └── style.css         # Custom CSS styles
```

## 🔧 Usage Guide

### Single Stock Analysis
1. Enter a stock ticker symbol (e.g., AAPL, MSFT)
2. Select date range using quick buttons or custom dates
3. Choose technical indicators from the sidebar
4. Click "🚀 Analyze Stock" to generate the analysis

### Portfolio Comparison
1. Switch to "Portfolio Comparison" mode
2. Enter multiple ticker symbols (one per line)
3. Select date range
4. Click "🔍 Compare Stocks" to see comparative analysis

### Popular Stock Categories
1. Select "Popular Picks" mode
2. Choose from categories: Tech Giants, Financial, Healthcare, etc.
3. Click on individual stocks or compare entire categories

## 📊 Technical Indicators

### Moving Averages
- **Simple Moving Average (SMA)**: 20, 50, 200-day periods
- **Exponential Moving Average (EMA)**: 12, 20, 26-day periods
- **Bollinger Bands**: 20-day period with 2 standard deviations

### Momentum Indicators
- **RSI (Relative Strength Index)**: 14-day period with overbought/oversold levels
- **MACD**: 12/26/9 configuration with histogram
- **Stochastic Oscillator**: %K and %D lines with 80/20 levels
- **Williams %R**: 14-day period momentum oscillator
- **ATR (Average True Range)**: 14-day volatility measure

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/test_stock_dashboard.py

# Or use pytest
python -m pytest tests/ -v
```

**Test Coverage:**
- ✅ Data fetching functionality
- ✅ Technical indicators calculations
- ✅ Utility functions
- ✅ Integration workflows
- ✅ Error handling
- ✅ Data validation

**Test Results:** 11/11 tests passing (100% success rate)

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

### Alternative Platforms
- **Heroku**: Full deployment guide in `DEPLOYMENT.md`
- **Railway**: One-click deployment
- **Docker**: Containerized deployment
- **AWS/Azure**: Cloud platform deployment

See `DEPLOYMENT.md` for detailed deployment instructions.

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit 1.50.0+
- **Data Processing**: Pandas, NumPy
- **Data Source**: yfinance API
- **Visualization**: Plotly 6.3.0+
- **Testing**: unittest
- **Styling**: Custom CSS

## 📈 Resume Impact

This project demonstrates:

### Technical Skills
- **Python Programming**: Clean, modular code with OOP principles
- **Data Analysis**: Financial data processing and statistical calculations
- **Web Development**: Interactive dashboard with modern UI/UX
- **API Integration**: Real-time data fetching and handling
- **Testing**: Unit testing and quality assurance
- **Version Control**: Git workflow and collaboration

### Domain Knowledge
- **Financial Analysis**: Understanding of technical indicators and market analysis
- **Data Visualization**: Creating meaningful, interactive charts
- **User Experience**: Intuitive interface design
- **Performance Optimization**: Caching and efficient data processing

### Software Engineering
- **Modular Architecture**: Separation of concerns and maintainable code
- **Error Handling**: Robust exception handling and user feedback
- **Documentation**: Comprehensive code documentation and user guides
- **Deployment**: Production-ready application deployment

## 🔮 Future Enhancements

### Phase 2 Features
- [ ] **Machine Learning**: Price prediction with LSTM/Prophet
- [ ] **Sentiment Analysis**: News and social media sentiment integration
- [ ] **Alerts System**: Email/SMS notifications for technical signals
- [ ] **Options Analysis**: Options chain data and Greeks calculations
- [ ] **Cryptocurrency**: Support for crypto trading pairs

### Phase 3 Features
- [ ] **Portfolio Tracking**: Real portfolio management features
- [ ] **Backtesting**: Historical strategy testing framework
- [ ] **API Development**: RESTful API for external integrations
- [ ] **Mobile App**: React Native or Flutter mobile application
- [ ] **Real-time Data**: WebSocket connections for live data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This dashboard is for educational and informational purposes only. It should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **Email**: your.email@domain.com

---

**⭐ If you found this project helpful, please give it a star on GitHub!**
