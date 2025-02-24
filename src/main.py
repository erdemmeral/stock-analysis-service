import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict
from .fundamental_analysis import FundamentalAnalyzer
from .technical_analysis import TechnicalAnalyzer
from .news_analysis import NewsAnalyzer
from .service.analysis_service import AnalysisService
import signal
import sys
from src.config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHANNEL_ID,
    TECHNICAL_ANALYSIS_INTERVAL,
    FUNDAMENTAL_ANALYSIS_INTERVAL,
    PORTFOLIO_THRESHOLD_SCORE,
    PORTFOLIO_API_URL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout
)
logger = logging.getLogger(__name__)

async def analyze_stocks() -> Dict:
    """Run fundamental, technical, and news analysis"""
    try:
        # Run fundamental analysis first
        logger.info("Starting Fundamental Analysis...")
        fund_analyzer = FundamentalAnalyzer()
        fund_results, raw_data = await fund_analyzer.analyze_stocks()
        
        logger.info(f"Found {len(fund_results)} stocks passing fundamental criteria")
        
        # Run technical analysis
        logger.info("Starting Technical Analysis...")
        tech_analyzer = TechnicalAnalyzer(timeframe='medium')
        
        medium_term_results = []
        long_term_results = []
        
        for stock in fund_results:
            try:
                ticker = stock['ticker']
                logger.info(f"Analyzing {ticker}")
                
                # Get medium term analysis
                med_analysis = tech_analyzer.analyze_stock(ticker)
                if 'error' not in med_analysis:
                    medium_term_results.append({
                        'ticker': ticker,
                        'fundamental_score': stock['score'],
                        'technical_analysis': med_analysis
                    })
                
                # Get long term analysis
                tech_analyzer.timeframe = 'long'
                long_analysis = tech_analyzer.analyze_stock(ticker)
                if 'error' not in long_analysis:
                    long_term_results.append({
                        'ticker': ticker,
                        'fundamental_score': stock['score'],
                        'technical_analysis': long_analysis
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
                continue
        
        # Add news analysis
        logger.info("Starting News Analysis...")
        news_analyzer = NewsAnalyzer()
        news_results = []
        
        for stock in fund_results:
            try:
                news_analysis = news_analyzer.analyze_stock_news(stock['ticker'])
                news_results.append(news_analysis)
            except Exception as e:
                logger.error(f"Error analyzing news for {stock['ticker']}: {str(e)}", exc_info=True)
                continue
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fundamental_results': fund_results,
            'raw_fundamental_data': raw_data,
            'medium_term_technical': medium_term_results,
            'long_term_technical': long_term_results,
            'news_analysis': news_results
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_stocks: {str(e)}", exc_info=True)
        raise

def calculate_total_score(fund_score: float, tech_score: float, news_score: float, timeframe: str) -> float:
    """Calculate weighted total score based on timeframe"""
    if timeframe == 'long':
        # Long-term weights: Fundamentals matter more
        weights = {
            'fundamental': 0.50,  # 50% weight
            'technical': 0.30,    # 30% weight
            'news': 0.20         # 20% weight
        }
    else:  # medium term
        # Medium-term weights: Technical and news matter more
        weights = {
            'fundamental': 0.30,  # 30% weight
            'technical': 0.40,    # 40% weight
            'news': 0.30         # 30% weight
        }
    
    return (fund_score * weights['fundamental'] + 
            tech_score * weights['technical'] + 
            news_score * weights['news'])

def print_top_stocks(results: List[Dict], timeframe: str):
    """Print formatted results with weighted total score"""
    print(f"\nTop Stocks ({timeframe.capitalize()} Term):")
    print("-" * 160)
    print(f"{'Ticker':<8} {'Fund Score':<12} {'Tech Score':<12} {'News Score':<12} "
          f"{'Total Score':<12} {'Trend':<12} {'Ichimoku':<12} {'Cloud':<12} "
          f"{'RSI':<12} {'MACD':<12} {'Prediction':<12}")
    print("-" * 160)
    
    # Get news scores from news_analysis results
    news_scores = {}
    if 'news_analysis' in results:
        for news_item in results['news_analysis']:
            if isinstance(news_item, dict) and 'ticker' in news_item:
                news_scores[news_item['ticker']] = news_item['news_score']
    
    # Calculate total scores and sort
    scored_results = []
    for stock in results:
        ticker = stock['ticker']
        tech_analysis = stock['technical_analysis']
        
        # Get technical indicators safely
        indicators = tech_analysis.get('signals', {}).get('indicators', {})
        ichimoku = tech_analysis.get('ichimoku', {}).get('signals', {})
        predictions = tech_analysis.get('price_predictions', {})
        
        total_score = calculate_total_score(
            stock['fundamental_score'],
            tech_analysis.get('technical_score', 0),
            news_scores.get(ticker, 50.0),
            timeframe
        )
        scored_results.append((stock, total_score))
    
    # Sort by total score
    sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
    
    # Print results
    for stock, total_score in sorted_results[:10]:  # Show top 10
        ticker = stock['ticker']
        tech = stock['technical_analysis']
        
        # Safely get all required values with defaults
        indicators = tech.get('signals', {}).get('indicators', {})
        ichimoku = tech.get('ichimoku', {}).get('signals', {})
        predictions = tech.get('price_predictions', {}).get('trend', {})
        
        # Get individual indicators with defaults
        ma_trend = indicators.get('moving_averages', {}).get('ma_trend', 'unknown')
        rsi_signal = indicators.get('momentum', {}).get('rsi_signal', 'unknown')
        macd_trend = indicators.get('trend', {}).get('macd_trend', 'unknown')
        
        # Get Ichimoku values with defaults
        ichimoku_signal = ichimoku.get('primary_signal', 'unknown')
        cloud_color = ichimoku.get('cloud_color', 'unknown')
        
        # Get prediction with default
        pred_change = predictions.get('predicted_change_percent', 0.0)
        
        print(f"{ticker:<8} "
              f"{stock['fundamental_score']:>8.1f}%    "
              f"{tech.get('technical_score', 0):>8.1f}%    "
              f"{news_scores.get(ticker, 50.0):>8.1f}%    "
              f"{total_score:>8.1f}%    "
              f"{ma_trend:<12} "
              f"{ichimoku_signal:<12} "
              f"{cloud_color:<12} "
              f"{rsi_signal:<12} "
              f"{macd_trend:<12} "
              f"{pred_change:>+8.1f}%")

    # Add summary of news sentiment
    print("\nNews Sentiment Summary:")
    print("-" * 80)
    for stock, _ in sorted_results[:5]:
        ticker = stock['ticker']
        if ticker in news_scores:
            news_data = next((item for item in results['news_analysis'] 
                            if item['ticker'] == ticker), None)
            if news_data:
                print(f"\n{ticker}:")
                print(f"Score: {news_scores[ticker]:.1f}% (Confidence: {news_data['confidence']})")
                print(f"Recent articles: {news_data['article_count']}")
                if news_data['sentiment_details']['recent_articles']:
                    print("Latest headlines:")
                    for article in news_data['sentiment_details']['recent_articles'][:3]:
                        print(f"- {article['date']}: {article['title']} "
                              f"(Sentiment: {article['sentiment']:.2f})")

def save_results(results: Dict):
    """Save analysis results to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_analysis_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {filename}")

async def shutdown(signal, loop, service):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    logger.error(f"Error in event loop: {msg}")

async def main():
    """Main function to run the analysis service"""
    try:
        service = AnalysisService()
        
        # Send startup message
        await service.send_telegram_alert("🚀 Stock Analysis Service Started")
        
        # Run initial fundamental analysis
        logger.info("Running initial fundamental analysis...")
        fund_analyzer = FundamentalAnalyzer()
        results, _ = await fund_analyzer.analyze_stocks()  # Now properly awaiting
        
        if results:
            logger.info(f"Initial fundamental analysis found {len(results)} stocks")
        
        # Start the main analysis loop
        await service.run_analysis_loop()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s\t\n%(levelname)s:%(name)s:%(message)s'
    )
    
    # Run the async main function
    asyncio.run(main())

async def test_main():
    """Test main program functionality"""
    try:
        service = AnalysisService()
        
        # Test 1: Configuration
        logger.info("Testing configuration...")
        from config import verify_config
        if not verify_config():
            logger.error("Configuration test failed")
            return False
            
        # Test 2: Watchlist loading
        logger.info("Testing watchlist loading...")
        watchlist = await service.get_watchlist()
        if not isinstance(watchlist, list):
            logger.error("Watchlist loading failed")
            return False
            
        # Test 3: Analysis pipeline
        logger.info("Testing analysis pipeline...")
        test_ticker = "AAPL"
        
        # Technical analysis
        tech_analyzer = service.tech_analyzer
        tech_analysis = tech_analyzer.analyze_stock(test_ticker)
        if not tech_analysis:
            logger.error("Technical analysis failed")
            return False
            
        # News analysis
        news_analyzer = service.news_analyzer
        news_analysis = news_analyzer.analyze_stock_news(test_ticker)
        if not news_analysis:
            logger.error("News analysis failed")
            return False
            
        # Test 4: Telegram notifications
        logger.info("Testing Telegram notifications...")
        try:
            await service.send_telegram_alert("🔧 Test message from stock analysis system")
            logger.info("Telegram notification test passed")
        except Exception as e:
            logger.error(f"Telegram notification test failed: {e}")
            return False
            
        logger.info("All main program tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Main program test failed: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Stock Analysis Service')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.test:
        # Run all tests
        logger.info("Running tests...")
        
        # Run technical analysis tests
        from .technical_analysis import run_component_tests
        tech_tests_passed = run_component_tests()
        
        # Run main program tests
        main_tests_passed = asyncio.run(test_main())
        
        if tech_tests_passed and main_tests_passed:
            logger.info("All tests passed successfully!")
            sys.exit(0)
        else:
            logger.error("Some tests failed!")
            sys.exit(1)
    else:
        # Run the main application
        logger.info("Starting stock analysis service...")
        asyncio.run(main())  # Changed: properly await the coroutine 