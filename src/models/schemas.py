from typing import Dict, Union, List, Any
from datetime import datetime
from pydantic import BaseModel

class WatchlistUpdate(BaseModel):
    last_analysis: datetime
    technical_score: float
    technical_scores: Dict[str, float]  # timeframe scores
    news_score: float
    news_sentiment: str
    risk_level: str

class TechnicalAnalysis(BaseModel):
    scores: Dict[str, Union[float, Dict[str, float]]]
    signals: Dict[str, Dict[str, Union[float, str]]]
    support_resistance: Dict[str, List[float]]

class NewsAnalysis(BaseModel):
    score: float
    sentiment: str
    confidence: str
    article_count: int
    recent_articles: List[Dict[str, Any]]

class CombinedAnalysis(BaseModel):
    ticker: str
    technical_analysis: TechnicalAnalysis
    news_analysis: NewsAnalysis
    timestamp: datetime 