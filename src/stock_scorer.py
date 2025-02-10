from typing import Dict
from .fundamental_analysis import FundamentalAnalyzer
from .technical_analysis import TechnicalAnalyzer
from .news_analysis import NewsAnalyzer

class StockScorer:
    def __init__(self):
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        
        # Weights for final score
        self.weights = {
            'fundamental': 0.4,
            'technical': 0.4,
            'news': 0.2
        }
    
    def get_total_score(self, ticker: str) -> Dict:
        """
        Calculates total score for a stock combining all three analyses
        """
        # Get individual scores
        fundamental_score = self.fundamental_analyzer.score_fundamentals(
            self.fundamental_analyzer.get_fundamental_data(ticker)
        )
        
        technical_score = self.technical_analyzer.score_technicals(
            self.technical_analyzer.get_technical_data(ticker)
        )
        
        news_score = self.news_analyzer.analyze_sentiment(
            self.news_analyzer.get_news(ticker)
        )
        
        # Calculate weighted score
        total_score = (
            fundamental_score * self.weights['fundamental'] +
            technical_score * self.weights['technical'] +
            news_score * self.weights['news']
        )
        
        return {
            'ticker': ticker,
            'total_score': total_score,
            'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'news_score': news_score
        } 