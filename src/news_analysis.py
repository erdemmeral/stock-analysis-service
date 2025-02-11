import yfinance as yf
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self):
        """Initialize FinBERT model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()  # Set to evaluation mode
        
    def get_full_article_text(self, url: str) -> str:
        """Get full text content from news article URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text based on common article containers
            article_content = soup.find(['article', 'main', 'div'], 
                                     {'class': ['article', 'content', 'article-content']})
            if article_content:
                return article_content.get_text(separator=' ', strip=True)
        
            return None
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
        return None

    def analyze_article_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of article text using FinBERT"""
        try:
            # Truncate text to max length
            max_length = 512
            inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, 
                                  truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get sentiment scores (positive, negative, neutral)
            scores = predictions[0].numpy()
            
            # Calculate weighted sentiment score (-1 to 1)
            sentiment_score = (scores[0] - scores[1]) * (1 - scores[2])
            
            return {
                'sentiment_score': float(sentiment_score),
                'positive': float(scores[0]),
                'negative': float(scores[1]),
                'neutral': float(scores[2])
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None

    def get_stock_news(self, ticker: str, days: int = 30) -> List[Dict]:
        """Get recent news articles for a stock"""
        try:
            # Use yf.Search instead of Ticker.news
            search = yf.Search(
                query=ticker,
                news_count=30,
                include_research=True
            )
            news = search.news
            
            # Filter for recent news
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_news = [
                article for article in news 
                if datetime.fromtimestamp(article['providerPublishTime']) > cutoff_date
            ]
            
            analyzed_news = []
            for article in recent_news:
                # Get full article text
                full_text = self.get_full_article_text(article['link'])
                if not full_text:
                    continue
                
                # Analyze sentiment
                sentiment = self.analyze_article_sentiment(full_text)
                if not sentiment:
                    continue
                
                analyzed_news.append({
                    'title': article['title'],
                    'date': datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d'),
                    'link': article['link'],
                    'sentiment': sentiment
                })
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return []

    def calculate_news_score(self, articles: List[Dict]) -> Dict:
        """Calculate overall news sentiment score with time-based weighting"""
        if not articles:
            return {
                'news_score': 50,  # Neutral score if no news
                'confidence': 'low',
                'article_count': 0,
                'weighted_sentiments': []
            }
        
        try:
            # Sort articles by date
            sorted_articles = sorted(articles, 
                                   key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
                                   reverse=True)
            
            # Calculate days from most recent article for each article
            most_recent = datetime.strptime(sorted_articles[0]['date'], '%Y-%m-%d')
            days_diff = [
                (most_recent - datetime.strptime(article['date'], '%Y-%m-%d')).days
                for article in sorted_articles
            ]
            
            # Calculate time-based weights
            # More recent articles get higher weights
            # Weight decay formula: w = exp(-days/decay_factor)
            decay_factor = 7  # Week-based decay
            time_weights = np.exp(-np.array(days_diff) / decay_factor)
            time_weights = time_weights / time_weights.sum()  # Normalize weights
            
            # Get sentiment scores
            sentiments = [article['sentiment']['sentiment_score'] for article in sorted_articles]
            
            # Calculate weighted sentiment score
            weighted_sentiments = []
            for i, (sentiment, weight) in enumerate(zip(sentiments, time_weights)):
                weighted_score = sentiment * weight
                weighted_sentiments.append({
                    'date': sorted_articles[i]['date'],
                    'title': sorted_articles[i]['title'],
                    'raw_sentiment': sentiment,
                    'weight': float(weight),
                    'weighted_score': float(weighted_score)
                })
            
            # Calculate final weighted average
            weighted_avg = sum(ws['weighted_score'] for ws in weighted_sentiments)
            
            # Convert to 0-100 score
            news_score = (weighted_avg + 1) * 50
            
            # Calculate confidence based on article count and sentiment consistency
            article_count = len(articles)
            sentiment_std = np.std(sentiments)
            
            # Adjust confidence based on time distribution
            time_span = max(days_diff)
            
            confidence = (
                'high' if (article_count >= 5 and sentiment_std < 0.3 and time_span <= 14) else
                'medium' if (article_count >= 3 and sentiment_std < 0.5 and time_span <= 30) else
                'low'
            )
            
            return {
                'news_score': float(news_score),
                'confidence': confidence,
                'article_count': article_count,
                'sentiment_std': float(sentiment_std),
                'time_span_days': time_span,
                'weighted_sentiments': weighted_sentiments[:5],  # Top 5 weighted articles
                'recent_articles': [{
                    'date': art['date'],
                    'title': art['title'],
                    'sentiment': art['sentiment']['sentiment_score'],
                    'link': art['link']
                } for art in sorted_articles[:5]]
            }
            
        except Exception as e:
            logger.error(f"Error calculating news score: {e}")
            return {
                'news_score': 50,
                'confidence': 'error',
                'article_count': len(articles),
                'error': str(e)
            }

    def analyze_stock_news(self, ticker: str) -> Dict:
        """Analyze news sentiment for a stock"""
        try:
            logger.info(f"Starting news analysis for {ticker}")
            news_items = self.get_stock_news(ticker)
            logger.info(f"Found {len(news_items)} news items for {ticker}")
            
            if not news_items:
                logger.warning(f"No news found for {ticker}")
                return {'news_score': 50, 'sentiment_details': {}}
            
            total_score = 0
            news_details = []
            
            for idx, news in enumerate(news_items, 1):
                # Analyze each news item
                sentiment = self.analyze_article_sentiment(news['title'] + ' ' + news['link'])['sentiment_score']
                relevance = self.calculate_relevance(news, ticker)
                recency = self.calculate_recency(news['date'])
                
                # Calculate weighted score for this news item
                item_score = sentiment * relevance * recency
                
                news_details.append({
                    'title': news['title'],
                    'date': news['date'],
                    'sentiment': sentiment,
                    'relevance': relevance,
                    'recency': recency,
                    'final_score': item_score
                })
                
                logger.info(f"News {idx}/{len(news_items)}:")
                logger.info(f"Title: {news['title']}")
                logger.info(f"Date: {news['date']}")
                logger.info(f"Sentiment: {sentiment:.2f}")
                logger.info(f"Relevance: {relevance:.2f}")
                logger.info(f"Recency: {recency:.2f}")
                logger.info(f"Item Score: {item_score:.2f}")
                
                total_score += item_score
            
            # Calculate final score (0-100)
            final_score = (total_score / len(news_items)) * 100
            final_score = max(0, min(100, final_score))  # Ensure between 0-100
            
            logger.info(f"\nFinal News Analysis for {ticker}:")
            logger.info(f"Total news items processed: {len(news_items)}")
            logger.info(f"Average sentiment score: {final_score:.2f}")
            logger.info("Top scoring news items:")
            
            # Sort and log top news items
            sorted_news = sorted(news_details, key=lambda x: x['final_score'], reverse=True)
            for idx, news in enumerate(sorted_news[:3], 1):
                logger.info(f"{idx}. Score: {news['final_score']:.2f} - {news['title']}")
            
            return {
                'news_score': final_score,
                'sentiment_details': {
                    'total_news': len(news_items),
                    'news_items': news_details,
                    'top_news': sorted_news[:3]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news for {ticker}: {e}")
            return {'news_score': 50, 'sentiment_details': {}} 