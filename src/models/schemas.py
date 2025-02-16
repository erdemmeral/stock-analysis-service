from typing import Dict, Union, List, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator

class WatchlistUpdate(BaseModel):
    last_analysis: datetime
    technical_score: float = Field(ge=0, le=100)
    technical_scores: Dict[str, float]  # timeframe scores
    news_score: float = Field(ge=0, le=100)
    news_sentiment: str
    risk_level: str
    current_price: float = Field(ge=0)
    fundamental_score: float = Field(ge=0, le=100, default=None)  # Make it optional with default None

    @validator('technical_scores')
    def validate_technical_scores(cls, v):
        for score in v.values():
            if not 0 <= score <= 100:
                raise ValueError('Technical scores must be between 0 and 100')
        return v

class TechnicalAnalysis(BaseModel):
    scores: Dict[str, Union[float, Dict[str, float]]]
    signals: Dict[str, Dict[str, Union[float, str]]]
    support_resistance: Dict[str, List[float]]

class NewsAnalysis(BaseModel):
    score: float = Field(ge=0, le=100)
    sentiment: str
    confidence: str
    article_count: int
    items: List[Dict[str, Any]]

class CombinedAnalysis(BaseModel):
    ticker: str
    technical_analysis: TechnicalAnalysis
    news_analysis: NewsAnalysis
    timestamp: datetime 