import unittest
from news_analysis import NewsAnalyzer
from datetime import datetime, timedelta
import numpy as np

class TestNewsAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = NewsAnalyzer()
        
    def test_sentiment_analysis(self):
        """Test sentiment analysis on sample texts"""
        test_cases = [
            {
                "text": "Company reports record profits and strong growth outlook",
                "expected_sentiment": "positive"
            },
            {
                "text": "Stock plummets after missing earnings expectations",
                "expected_sentiment": "negative"
            },
            {
                "text": "Company announces regular quarterly dividend",
                "expected_sentiment": "neutral"
            }
        ]
        
        for case in test_cases:
            sentiment = self.analyzer.analyze_article_sentiment(case["text"])
            self.assertIsNotNone(sentiment)
            self.assertIn('sentiment_score', sentiment)
            
            # Check if sentiment aligns with expectation
            if case["expected_sentiment"] == "positive":
                self.assertGreater(sentiment['sentiment_score'], 0)
            elif case["expected_sentiment"] == "negative":
                self.assertLess(sentiment['sentiment_score'], 0)
    
    def test_news_retrieval(self):
        """Test news retrieval for well-known stocks"""
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        for ticker in test_tickers:
            news = self.analyzer.get_stock_news(ticker)
            self.assertIsInstance(news, list)
            
            if news:  # If news found
                self.assertGreater(len(news), 0)
                first_article = news[0]
                
                # Check article structure
                self.assertIn('title', first_article)
                self.assertIn('date', first_article)
                self.assertIn('link', first_article)
                self.assertIn('sentiment', first_article)
    
    def test_news_scoring(self):
        """Test news scoring with mock articles"""
        mock_articles = [
            {
                'title': 'Test Article 1',
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'link': 'http://test1.com',
                'sentiment': {'sentiment_score': 0.8}
            },
            {
                'title': 'Test Article 2',
                'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'link': 'http://test2.com',
                'sentiment': {'sentiment_score': 0.6}
            },
            {
                'title': 'Test Article 3',
                'date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'link': 'http://test3.com',
                'sentiment': {'sentiment_score': -0.3}
            }
        ]
        
        score = self.analyzer.calculate_news_score(mock_articles)
        
        # Check score structure
        self.assertIn('news_score', score)
        self.assertIn('confidence', score)
        self.assertIn('article_count', score)
        self.assertIn('weighted_sentiments', score)
        
        # Check score ranges
        self.assertGreaterEqual(score['news_score'], 0)
        self.assertLessEqual(score['news_score'], 100)
        self.assertEqual(score['article_count'], 3)
    
    def test_full_analysis(self):
        """Test complete news analysis for a stock"""
        test_ticker = 'AAPL'  # Use Apple as it usually has news
        
        result = self.analyzer.analyze_stock_news(test_ticker)
        
        # Check result structure
        self.assertIn('ticker', result)
        self.assertIn('news_score', result)
        self.assertIn('confidence', result)
        self.assertIn('article_count', result)
        self.assertIn('sentiment_details', result)
        self.assertIn('timestamp', result)
        
        # Check score ranges
        self.assertGreaterEqual(result['news_score'], 0)
        self.assertLessEqual(result['news_score'], 100)
        
        # Check confidence levels
        self.assertIn(result['confidence'], ['low', 'medium', 'high', 'error'])

def main():
    # Run tests
    print("\nStarting News Analysis Tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    main() 