from fastapi import APIRouter, HTTPException
from datetime import datetime
from ..technical_analysis import TechnicalAnalyzer
from ..news_analysis import NewsAnalyzer
from ..models.schemas import WatchlistUpdate
import yfinance as yf
import aiohttp
from ..config import PORTFOLIO_API_URL

router = APIRouter()

@router.get("/analysis/{ticker}")
async def get_analysis(ticker: str):
    """Get combined analysis for a ticker"""
    try:
        tech_analyzer = TechnicalAnalyzer()
        news_analyzer = NewsAnalyzer()
        
        # Get analyses
        tech_analysis = tech_analyzer.analyze_stock(ticker)
        news_analysis = news_analyzer.analyze_stock_news(ticker)
        
        # Get current price
        current_price = tech_analysis.get('signals', {}).get('current_price')
        if current_price is None:
            try:
                stock_info = yf.Ticker(ticker).info
                current_price = stock_info.get('regularMarketPrice', 0.0)
            except:
                current_price = 0.0
        
        return {
            "ticker": ticker,
            "technical_analysis": {
                "scores": {
                    "total": tech_analysis['technical_score']['total'],
                    "timeframes": tech_analysis['technical_score']['timeframes'],
                    "components": {
                        "momentum": tech_analysis['technical_score']['momentum'],
                        "trend": tech_analysis['technical_score']['trend'],
                        "volume": tech_analysis['technical_score']['volume']
                    }
                },
                "signals": {
                    "trend": tech_analysis['signals']['trend'],
                    "momentum": tech_analysis['signals']['momentum'],
                    "volume": tech_analysis['signals']['volume'],
                    "volatility": tech_analysis['signals']['volatility']
                },
                "support_resistance": tech_analysis['support_resistance']
            },
            "news_analysis": {
                "score": news_analysis['news_score'],
                "sentiment": news_analysis['sentiment'],
                "confidence": news_analysis['confidence'],
                "article_count": news_analysis['article_count'],
                "items": news_analysis['items']
            },
            "current_price": current_price
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/watchlist/{ticker}")
async def update_watchlist_item(ticker: str, update_data: WatchlistUpdate):
    """Update watchlist item with new analysis"""
    try:
        # Updated schema to match new technical analysis structure
        watchlist_data = {
            "last_analysis": update_data.last_analysis,
            "technical_score": update_data.technical_score,
            "technical_scores": update_data.technical_scores,  # Single set of timeframe scores
            "news_score": update_data.news_score,
            "news_sentiment": update_data.news_sentiment,
            "risk_level": update_data.risk_level,
            "current_price": update_data.current_price,
            "replace_scores": True  # Always replace scores instead of adding new ones
        }
        
        # Update using Portfolio API
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                f"{PORTFOLIO_API_URL}/watchlist/{ticker}",
                json=watchlist_data
            ) as response:
                if response.status == 404:
                    # If item doesn't exist, create it
                    async with session.post(
                        f"{PORTFOLIO_API_URL}/watchlist",
                        json={"ticker": ticker, **watchlist_data}
                    ) as create_response:
                        if create_response.status not in (200, 201):
                            raise HTTPException(
                                status_code=create_response.status,
                                detail="Failed to create watchlist item"
                            )
                elif response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to update watchlist item"
                    )
                
        return {"status": "success", "message": "Watchlist item updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 