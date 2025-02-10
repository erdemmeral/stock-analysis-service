import logging
from typing import Dict, List
import telegram
from datetime import datetime
import asyncio
from ..technical_analysis import TechnicalAnalyzer
from ..database import Database  # Your backend database connection
from ..config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHANNEL_ID,
    ANALYSIS_INTERVAL,
    PORTFOLIO_THRESHOLD_SCORE
)

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        self.db = Database()  # Your backend database connection
        
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

    async def analyze_and_notify(self, ticker: str):
        """Analyze single stock and send notifications if needed"""
        try:
            # Get analysis
            analysis = self.tech_analyzer.analyze_stock(ticker)
            
            # Check if analysis indicates strong buy/sell
            tech_score = analysis['technical_score']['total_score']
            
            # Format message
            if tech_score >= 70:  # Strong buy
                await self.handle_buy_signal(ticker, analysis)
            elif tech_score <= 30:  # Strong sell
                await self.handle_sell_signal(ticker, analysis)
                
            # Store analysis results
            await self.store_analysis(ticker, analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")

    async def handle_buy_signal(self, ticker: str, analysis: Dict):
        """Handle buy signals"""
        try:
            # Check if already in portfolio
            if not await self.db.is_in_portfolio(ticker):
                # Format buy alert
                message = (
                    f"🚨 <b>STRONG BUY Signal</b> 🚨\n"
                    f"Ticker: {ticker}\n"
                    f"Technical Score: {analysis['technical_score']['total_score']:.1f}\n"
                    f"Current Price: ${analysis['signals']['current_price']:.2f}\n"
                    f"Expected Move: {analysis['predictions']['final']['predicted_change_percent']:.1f}%\n"
                    f"Confidence: {analysis['predictions']['final']['confidence']:.2f}\n"
                    f"Support Levels: ${analysis['support_resistance']['support']['levels'][0]:.2f}"
                )
                
                # Send alert
                await self.send_telegram_alert(message)
                
                # Add to portfolio if meets criteria
                if analysis['technical_score']['total_score'] >= PORTFOLIO_THRESHOLD_SCORE:
                    await self.db.add_to_portfolio(ticker, analysis)
                
        except Exception as e:
            logger.error(f"Error handling buy signal for {ticker}: {e}")

    async def handle_sell_signal(self, ticker: str, analysis: Dict):
        """Handle sell signals"""
        try:
            # Check if in portfolio
            if await self.db.is_in_portfolio(ticker):
                # Format sell alert
                message = (
                    f"🔴 <b>STRONG SELL Signal</b> 🔴\n"
                    f"Ticker: {ticker}\n"
                    f"Technical Score: {analysis['technical_score']['total_score']:.1f}\n"
                    f"Current Price: ${analysis['signals']['current_price']:.2f}\n"
                    f"Expected Move: {analysis['predictions']['final']['predicted_change_percent']:.1f}%\n"
                    f"Confidence: {analysis['predictions']['final']['confidence']:.2f}\n"
                    f"Resistance Levels: ${analysis['support_resistance']['resistance']['levels'][0]:.2f}"
                )
                
                # Send alert
                await self.send_telegram_alert(message)
                
                # Remove from portfolio
                await self.db.remove_from_portfolio(ticker)
                
        except Exception as e:
            logger.error(f"Error handling sell signal for {ticker}: {e}")

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

    async def run_analysis_loop(self):
        """Main analysis loop"""
        while True:
            try:
                # Get watchlist from database
                watchlist = await self.db.get_watchlist()
                
                # Analyze each stock
                for ticker in watchlist:
                    await self.analyze_and_notify(ticker)
                    
                # Wait for next interval
                await asyncio.sleep(ANALYSIS_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying 