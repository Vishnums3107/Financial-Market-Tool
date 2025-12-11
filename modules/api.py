"""
REST API Module

FastAPI backend for analysis endpoints:
- Stock analysis endpoints
- Trading signals
- Backtesting
- Webhooks
"""

import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_api_app():
    """
    Create FastAPI application.
    
    Returns:
        FastAPI app instance
    """
    try:
        from fastapi import FastAPI, HTTPException, Query, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="Stock Analysis API",
        description="REST API for stock analysis and trading signals",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request models
    class AnalysisRequest(BaseModel):
        symbol: str
        start_date: Optional[str] = None
        end_date: Optional[str] = None
        indicators: Optional[List[str]] = []
    
    class BacktestRequest(BaseModel):
        symbol: str
        strategy: str
        start_date: str
        end_date: str
        initial_capital: float = 10000
        risk_per_trade: float = 0.02
    
    class AlertRequest(BaseModel):
        symbol: str
        alert_type: str
        condition_value: float
        message: Optional[str] = ""
    
    class WebhookPayload(BaseModel):
        event: str
        symbol: str
        data: dict
    
    # ==================== ENDPOINTS ====================
    
    @app.get("/")
    async def root():
        return {
            "message": "Stock Analysis API",
            "version": "1.0.0",
            "endpoints": [
                "/analyze/{symbol}",
                "/signals/{symbol}",
                "/price-action/{symbol}",
                "/backtest",
                "/quote/{symbol}"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/quote/{symbol}")
    async def get_quote(symbol: str):
        """Get current quote for a symbol."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            return {
                "symbol": symbol.upper(),
                "price": info.get("regularMarketPrice", 0),
                "change": info.get("regularMarketChange", 0),
                "change_pct": info.get("regularMarketChangePercent", 0),
                "volume": info.get("volume", 0),
                "high": info.get("dayHigh", 0),
                "low": info.get("dayLow", 0),
                "open": info.get("open", 0),
                "prev_close": info.get("previousClose", 0)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/analyze/{symbol}")
    async def analyze_stock(
        symbol: str,
        days: int = Query(365, description="Number of days of history"),
        indicators: str = Query("", description="Comma-separated indicators")
    ):
        """Get full stock analysis."""
        try:
            from modules.data_fetcher import fetch_stock_data
            from modules.indicators import calculate_all_indicators
            from modules.price_action import analyze_price_action
            
            end = date.today()
            start = end - timedelta(days=days)
            
            data = fetch_stock_data(symbol.upper(), start, end)
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found")
            
            # Add indicators
            indicator_list = indicators.split(",") if indicators else []
            if indicator_list:
                data = calculate_all_indicators(data, indicator_list)
            
            # Price action analysis
            analyzed_data, sr_data, summary = analyze_price_action(data)
            
            return {
                "symbol": symbol.upper(),
                "period": {"start": str(start), "end": str(end)},
                "current_price": float(data['Close'].iloc[-1]),
                "price_action": summary,
                "support_levels": sr_data.get('support_levels', [])[:3],
                "resistance_levels": sr_data.get('resistance_levels', [])[:3]
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/signals/{symbol}")
    async def get_signals(
        symbol: str,
        strategy: str = Query("level_signal_rr", description="Strategy name"),
        days: int = Query(365, description="Days of history")
    ):
        """Get trading signals for a symbol."""
        try:
            from modules.data_fetcher import fetch_stock_data
            from modules.strategies import generate_trading_signals
            
            end = date.today()
            start = end - timedelta(days=days)
            
            data = fetch_stock_data(symbol.upper(), start, end)
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found")
            
            analyzed_data, sr_data, signal_summary = generate_trading_signals(
                data, strategy
            )
            
            # Get recent signals
            signals = []
            if 'Signal' in analyzed_data.columns:
                signal_rows = analyzed_data[analyzed_data['Signal'] != ''].tail(10)
                for idx, row in signal_rows.iterrows():
                    signals.append({
                        "date": str(idx)[:10],
                        "signal": row['Signal'],
                        "entry": float(row.get('Entry', row['Close'])),
                        "stop_loss": float(row.get('StopLoss', 0)),
                        "take_profit": float(row.get('TakeProfit', 0))
                    })
            
            return {
                "symbol": symbol.upper(),
                "strategy": strategy,
                "signals": signals,
                "summary": signal_summary
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/price-action/{symbol}")
    async def get_price_action(
        symbol: str,
        days: int = Query(365, description="Days of history")
    ):
        """Get price action analysis."""
        try:
            from modules.data_fetcher import fetch_stock_data
            from modules.price_action import analyze_price_action
            from modules.ml_patterns import analyze_patterns
            
            end = date.today()
            start = end - timedelta(days=days)
            
            data = fetch_stock_data(symbol.upper(), start, end)
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found")
            
            analyzed_data, sr_data, pa_summary = analyze_price_action(data)
            pattern_analysis = analyze_patterns(data)
            
            return {
                "symbol": symbol.upper(),
                "trend": pa_summary.get('trend', 'Unknown'),
                "phase": pa_summary.get('phase', 'Unknown'),
                "support_resistance": sr_data,
                "patterns": {
                    "found": pattern_analysis.get('patterns_found', 0),
                    "bias": pattern_analysis.get('bias', 'Neutral'),
                    "top_pattern": pattern_analysis.get('top_pattern')
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/backtest")
    async def run_backtest(request: BacktestRequest):
        """Run a backtest."""
        try:
            from modules.data_fetcher import fetch_stock_data
            from modules.strategies import generate_trading_signals
            from modules.backtesting import BacktestEngine, BacktestConfig
            
            start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
            end = datetime.strptime(request.end_date, "%Y-%m-%d").date()
            
            data = fetch_stock_data(request.symbol.upper(), start, end)
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found")
            
            # Generate signals
            analyzed_data, _, _ = generate_trading_signals(data, request.strategy)
            
            # Run backtest
            config = BacktestConfig(
                initial_capital=request.initial_capital,
                risk_per_trade_pct=request.risk_per_trade
            )
            engine = BacktestEngine(config)
            results = engine.run_backtest(analyzed_data, analyzed_data)
            
            return {
                "symbol": request.symbol.upper(),
                "strategy": request.strategy,
                "period": {"start": request.start_date, "end": request.end_date},
                "metrics": results['metrics'],
                "trades_count": len(results['trades'])
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/forecast/{symbol}")
    async def get_forecast(
        symbol: str,
        horizon: int = Query(5, description="Forecast horizon in days")
    ):
        """Get price forecast."""
        try:
            from modules.data_fetcher import fetch_stock_data
            from modules.ml_forecasting import forecast_price
            
            end = date.today()
            start = end - timedelta(days=365)
            
            data = fetch_stock_data(symbol.upper(), start, end)
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found")
            
            result = forecast_price(data, horizon)
            
            return {
                "symbol": symbol.upper(),
                "method": result.method,
                "confidence": result.confidence,
                "forecast": [
                    {
                        "date": str(result.dates[i])[:10],
                        "price": float(result.predictions[i]),
                        "lower": float(result.lower_bound[i]),
                        "upper": float(result.upper_bound[i])
                    }
                    for i in range(len(result.predictions))
                ]
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/webhook/signal")
    async def receive_webhook(payload: WebhookPayload):
        """Receive webhook notification."""
        # Log webhook
        print(f"Webhook received: {payload.event} for {payload.symbol}")
        
        # Here you could trigger actions based on the webhook
        return {
            "status": "received",
            "event": payload.event,
            "symbol": payload.symbol,
            "timestamp": datetime.now().isoformat()
        }
    
    return app


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    try:
        import uvicorn
        app = create_api_app()
        if app:
            uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")


# Create app instance for direct uvicorn use
app = create_api_app()


if __name__ == "__main__":
    run_api()
