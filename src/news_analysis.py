import yfinance as yf
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
        """Initialize FinBERT-tone model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.labels = ['negative', 'neutral', 'positive']
        self.model.eval()  # Set to evaluation mode
        
    def analyze_article_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of article text using FinBERT-tone"""
        try:
            # Get model outputs
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Convert to numpy for easier handling
            probs = probabilities.detach().numpy()[0]
            
            # Map sentiment scores:
            # negative: 0-35
            # neutral: 35-65
            # positive: 65-100
            sentiment_mapping = {
                0: 25,    # negative center
                1: 50,    # neutral center
                2: 75     # positive center
            }
            
            # Get predicted class and confidence
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            # Calculate final sentiment score (0-100)
            base_score = sentiment_mapping[predicted_class]
            score_adjustment = (confidence - 0.33) * 30  # Adjust based on confidence
            final_score = base_score + score_adjustment
            
            # Ensure score is in 0-100 range
            final_score = max(0, min(100, final_score))
            
            return {
                'sentiment_score': final_score,
                'label': 'negative' if final_score < 35 else 'positive' if final_score > 65 else 'neutral',
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
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
                try:
                    # Analyze sentiment using only the title
                    sentiment = self.analyze_article_sentiment(article['title'])
                    if not sentiment:
                        continue
                    
                    analyzed_news.append({
                        'title': article['title'],
                        'date': datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d'),
                        'source': article.get('publisher', ''),
                        'sentiment': sentiment
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing article {article['title']}: {e}")
                    continue
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return []

    def calculate_relevance(self, news_item: Dict) -> float:
        """Calculate relevance score for a news item"""
        try:
            # Base relevance starts at 1.0
            relevance = 1.0
            
            # Check title keywords
            title = news_item.get('title', '').lower()
            important_keywords = {
                'earnings': 1.5,
                'revenue': 1.3,
                'profit': 1.3,
                'guidance': 1.2,
                'upgrade': 1.4,
                'downgrade': 1.4,
                'acquisition': 1.5,
                'merger': 1.5,
                'lawsuit': 1.3,
                'fda': 1.4,
                'patent': 1.3,
                'contract': 1.3
            }
            
            # Adjust relevance based on keywords
            for keyword, multiplier in important_keywords.items():
                if keyword in title:
                    relevance *= multiplier
            
            # Time decay factor (newer news is more relevant)
            news_date = datetime.fromisoformat(news_item.get('date', datetime.now().isoformat()))
            days_old = (datetime.now() - news_date).days
            time_decay = max(0.5, 1.0 - (days_old * 0.1))  # Minimum 0.5 relevance
            relevance *= time_decay
            
            # Source credibility factor
            source = news_item.get('source', '').lower()
            credible_sources = {
                'reuters': 1.3,
                'bloomberg': 1.3,
                'wsj': 1.3,
                'cnbc': 1.2,
                'marketwatch': 1.2,
                'fool.com': 0.9,
                'seekingalpha': 0.9,
                'benzinga': 0.9
            }
            source_factor = credible_sources.get(source, 1.0)
            relevance *= source_factor
            
            # Cap maximum relevance at 2.0
            return min(2.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 1.0

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
            
            # Get sentiment scores and ensure they're in 0-100 range
            sentiments = []
            for article in sorted_articles:
                score = article['sentiment']['sentiment_score']
                # Clamp sentiment scores to 0-100 range
                score = max(0, min(100, score))
                sentiments.append(score)
            
            # Calculate weighted sentiment score
            weighted_sentiments = []
            total_weighted_score = 0
            total_weight = 0
            
            for i, (sentiment, weight) in enumerate(zip(sentiments, time_weights)):
                weighted_score = sentiment * weight
                total_weighted_score += weighted_score
                total_weight += weight
                
                weighted_sentiments.append({
                    'date': sorted_articles[i]['date'],
                    'title': sorted_articles[i]['title'],
                    'raw_sentiment': sentiment,
                    'weight': float(weight),
                    'weighted_score': float(weighted_score)
                })
            
            # Calculate final weighted average properly
            news_score = total_weighted_score / total_weight if total_weight > 0 else 50
            
            # Ensure the score is in 0-100 range
            news_score = max(0, min(100, news_score))
            
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
            
            # Log the calculation details for debugging
            logger.debug(f"News score calculation details:")
            logger.debug(f"Total weighted score: {total_weighted_score}")
            logger.debug(f"Total weight: {total_weight}")
            logger.debug(f"Final news score: {news_score}")
            
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
                    'sentiment': max(0, min(100, art['sentiment']['sentiment_score'])),  # Ensure sentiment is bounded
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
        """Analyze news for a given stock"""
        try:
            logger.info(f"Starting news analysis for {ticker}")
            
            # Get news data
            news_items = self.get_stock_news(ticker)
            if not news_items:
                logger.warning(f"No news found for {ticker}")
                return {
                    'news_score': 50,
                    'sentiment': 'neutral',
                    'items': [],
                    'confidence': 'low',
                    'article_count': 0
                }
            
            logger.info(f"Found {len(news_items)} news items for {ticker}")
            
            # Process each news item
            processed_items = []
            total_sentiment = 0
            total_relevance = 0
            
            for item in news_items:
                # Get sentiment from pre-analyzed news
                sentiment_data = item['sentiment']
                
                # Log sentiment details for debugging
                logger.info(f"Article: {item['title']}")
                logger.info(f"Sentiment Label: {sentiment_data['label']}")
                logger.info(f"Sentiment Score: {sentiment_data['sentiment_score']}")
                
                # Calculate relevance
                relevance = self.calculate_relevance(item)
                
                # Weight sentiment by relevance
                total_sentiment += sentiment_data['sentiment_score'] * relevance
                total_relevance += relevance
                
                processed_items.append({
                    'title': item['title'],
                    'date': item['date'],
                    'source': item['source'],
                    'sentiment': sentiment_data['sentiment_score'],
                    'sentiment_label': sentiment_data['label'],
                    'confidence': sentiment_data['confidence'],
                    'relevance': relevance
                })
            
            # Sort by relevance and recency
            processed_items.sort(key=lambda x: (x['relevance'], x['date']), reverse=True)
            
            # Calculate final weighted sentiment score (0-100)
            if total_relevance > 0:
                weighted_sentiment = (total_sentiment / total_relevance)
                news_score = 50 + (weighted_sentiment * 25)  # Convert to 0-100 scale
            else:
                news_score = 50
            
            # Updated sentiment thresholds
            if news_score >= 60:  # Lowered from 65
                sentiment = 'positive'
            elif news_score <= 40:  # Raised from 35
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            logger.info(f"News analysis complete for {ticker} - Score: {news_score:.2f}, Sentiment: {sentiment}")
            
            return {
                'news_score': float(news_score),
                'sentiment': sentiment,
                'items': processed_items[:5],
                'confidence': 'high' if len(processed_items) >= 5 else 'medium' if len(processed_items) >= 3 else 'low',
                'article_count': len(processed_items)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news for {ticker}: {e}")
            return {
                'news_score': 50,
                'sentiment': 'neutral',
                'items': [],
                'confidence': 'low',
                'article_count': 0
            }

def run_test_analysis():
    """Test function to verify news analysis"""
    logger.info("Starting news analysis test...")
    
    # Initialize analyzer
    analyzer = NewsAnalyzer()
    
    # Test with a well-known stock
    test_ticker = "AAPL"
    
    try:
        # Get and analyze news
        analysis = analyzer.analyze_stock_news(test_ticker)
        
        print("\n=== News Analysis Test Results ===")
        print(f"Stock: {test_ticker}")
        
        print("\n1. Overall Metrics:")
        print(f"News Score: {analysis['news_score']:.2f}")
        print(f"Overall Sentiment: {analysis['sentiment']}")
        
        print("\n2. Recent Articles:")
        for item in analysis['items']:
            print(f"\nTitle: {item['title']}")
            print(f"Date: {item['date']}")
            print(f"Source: {item['source']}")
            print(f"Sentiment Label: {item['sentiment_label']}")
            print(f"Sentiment Score: {item['sentiment']:.2f}")
            print(f"Confidence: {item['confidence']:.2f}")
            print(f"Relevance Score: {item['relevance']:.2f}")
        
        print("\n=== Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_test_analysis() 