from fastapi import APIRouter, HTTPException
from datetime import datetime
from src.service.technical_analyzer import TechnicalAnalyzer
from src.service.news_analyzer import NewsAnalyzer
from src.models.schemas import WatchlistUpdate
from src.database.database import db

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
                "recent_articles": news_analysis['items']
            },
            "timestamp": datetime.now().isoformat()
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
            "technical_scores": update_data.technical_scores,  # New field for timeframe scores
            "news_score": update_data.news_score,
            "news_sentiment": update_data.news_sentiment,
            "risk_level": update_data.risk_level
        }
        
        # Update database
        result = await db.watchlist.update_one(
            {"ticker": ticker},
            {"$set": watchlist_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Watchlist item not found")
            
        return {"status": "success", "message": "Watchlist item updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 