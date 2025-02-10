import logging
from typing import Dict, List
import telegram
from datetime import datetime, timedelta
import asyncio
from ..technical_analysis import TechnicalAnalyzer
from ..fundamental_analysis import FundamentalAnalyzer
from ..news_analysis import NewsAnalyzer
from ..database import Database  # Your backend database connection
from ..config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHANNEL_ID,
    TECHNICAL_ANALYSIS_INTERVAL,  # e.g. 3600 (1 hour)
    FUNDAMENTAL_ANALYSIS_INTERVAL,  # 86400 (24 hours)
    PORTFOLIO_THRESHOLD_SCORE
)

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        self.fund_analyzer = FundamentalAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        self.db = Database()  # Your backend database connection
        self.last_fundamental_run = None
        
    async def send_telegram_alert(self, message: str):
        """Send alert to Telegram channel"""
        try:
            await self.bot.send_message(
                chat_id=TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def run_fundamental_analysis(self):
        """Run fundamental analysis to update watchlist"""
        try:
            logger.info("Running fundamental analysis...")
            fund_results, raw_data = self.fund_analyzer.analyze_stocks()
            
            # Update watchlist in database
            await self.db.update_watchlist(fund_results)
            
            # Store raw fundamental data
            await self.db.store_fundamental_data(raw_data)
            
            logger.info(f"Updated watchlist with {len(fund_results)} stocks")
            return fund_results
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return []

    async def analyze_watchlist(self):
        """Run technical and news analysis on watchlist stocks"""
        try:
            # Get current watchlist
            watchlist = await self.db.get_watchlist()
            
            for ticker in watchlist:
                try:
                    # Get fundamental score from database
                    fund_score = await self.db.get_fundamental_score(ticker)
                    
                    # Run technical analysis
                    tech_analysis = self.tech_analyzer.analyze_stock(ticker)
                    tech_score = tech_analysis['technical_score']['total_score']
                    
                    # Run news analysis
                    news_analysis = self.news_analyzer.analyze_stock_news(ticker)
                    news_score = news_analysis['news_score']
                    
                    # Calculate combined score
                    combined_score = self.calculate_combined_score(
                        fund_score, tech_score, news_score
                    )
                    
                    # Check if stock should be in portfolio
                    if combined_score >= PORTFOLIO_THRESHOLD_SCORE:
                        if not await self.db.is_in_portfolio(ticker):
                            await self.handle_portfolio_addition(
                                ticker, 
                                combined_score,
                                tech_analysis,
                                news_analysis
                            )
                    else:
                        if await self.db.is_in_portfolio(ticker):
                            await self.handle_portfolio_removal(
                                ticker,
                                combined_score,
                                tech_analysis,
                                news_analysis
                            )
                    
                    # Store analysis results
                    await self.store_analysis(
                        ticker,
                        fund_score,
                        tech_analysis,
                        news_analysis,
                        combined_score
                    )
                    
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in watchlist analysis: {e}")

    def calculate_combined_score(
        self, 
        fund_score: float, 
        tech_score: float, 
        news_score: float
    ) -> float:
        """Calculate weighted combined score"""
        weights = {
            'fundamental': 0.4,  # 40% weight
            'technical': 0.4,    # 40% weight
            'news': 0.2         # 20% weight
        }
        
        return (
            fund_score * weights['fundamental'] +
            tech_score * weights['technical'] +
            news_score * weights['news']
        )

    async def run_analysis_loop(self):
        """Main analysis loop with different intervals"""
        while True:
            try:
                current_time = datetime.now()
                
                # Run fundamental analysis every 24 hours
                if (self.last_fundamental_run is None or 
                    (current_time - self.last_fundamental_run).total_seconds() >= FUNDAMENTAL_ANALYSIS_INTERVAL):
                    await self.run_fundamental_analysis()
                    self.last_fundamental_run = current_time
                
                # Run technical and news analysis on watchlist
                await self.analyze_watchlist()
                
                # Wait for next technical analysis interval
                await asyncio.sleep(TECHNICAL_ANALYSIS_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def handle_portfolio_addition(
        self,
        ticker: str,
        score: float,
        tech_analysis: Dict,
        news_analysis: Dict
    ):
        """Handle adding stock to portfolio"""
        try:
            # Add to portfolio
            await self.db.add_to_portfolio(ticker, {
                'score': score,
                'entry_price': tech_analysis['signals']['current_price'],
                'entry_date': datetime.now(),
                'analysis': {
                    'technical': tech_analysis,
                    'news': news_analysis
                }
            })
            
            # Send notification
            message = (
                f"🚨 <b>New Portfolio Addition</b> 🚨\n"
                f"Ticker: {ticker}\n"
                f"Combined Score: {score:.1f}\n"
                f"Entry Price: ${tech_analysis['signals']['current_price']:.2f}\n"
                f"Technical Score: {tech_analysis['technical_score']['total_score']:.1f}\n"
                f"News Sentiment: {news_analysis['sentiment']}\n"
                f"Expected Move: {tech_analysis['predictions']['final']['predicted_change_percent']:.1f}%"
            )
            
            await self.send_telegram_alert(message)
            
        except Exception as e:
            logger.error(f"Error adding {ticker} to portfolio: {e}")

    async def handle_portfolio_removal(
        self,
        ticker: str,
        score: float,
        tech_analysis: Dict,
        news_analysis: Dict
    ):
        """Handle removing stock from portfolio"""
        try:
            # Get entry data
            entry_data = await self.db.get_portfolio_entry(ticker)
            entry_price = entry_data['entry_price']
            current_price = tech_analysis['signals']['current_price']
            
            # Calculate return
            returns_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Remove from portfolio
            await self.db.remove_from_portfolio(ticker)
            
            # Send notification
            message = (
                f"🔴 <b>Portfolio Exit</b> 🔴\n"
                f"Ticker: {ticker}\n"
                f"Exit Price: ${current_price:.2f}\n"
                f"Return: {returns_pct:+.1f}%\n"
                f"Current Score: {score:.1f}\n"
                f"Technical Signal: {tech_analysis['predictions']['final']['direction']}\n"
                f"News Sentiment: {news_analysis['sentiment']}"
            )
            
            await self.send_telegram_alert(message)
            
        except Exception as e:
            logger.error(f"Error removing {ticker} from portfolio: {e}")

    async def store_analysis(self, ticker: str, analysis: Dict):
        """Store analysis results in database"""
        try:
            await self.db.store_analysis_result(
                ticker=ticker,
                timestamp=datetime.now(),
                analysis_data=analysis
            )
        except Exception as e:
            logger.error(f"Error storing analysis for {ticker}: {e}") 