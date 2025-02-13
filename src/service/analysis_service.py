import logging
from typing import Dict, List
import telegram
from datetime import datetime
import asyncio
import aiohttp
from ..technical_analysis import TechnicalAnalyzer
from ..fundamental_analysis import FundamentalAnalyzer
from ..news_analysis import NewsAnalyzer
from ..config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHANNEL_ID,
    TECHNICAL_ANALYSIS_INTERVAL,
    FUNDAMENTAL_ANALYSIS_INTERVAL,
    PORTFOLIO_THRESHOLD_SCORE,
    PORTFOLIO_API_URL,
    PROFIT_TARGETS,
    TRAILING_STOP
)
import yfinance as yf

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        self.fund_analyzer = FundamentalAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        self.last_fundamental_run = None
        
        # Add configuration properties
        self.analysis_interval = TECHNICAL_ANALYSIS_INTERVAL
        self.portfolio_threshold = PORTFOLIO_THRESHOLD_SCORE
        self.watchlist = self._load_watchlist()

    def _load_watchlist(self) -> List[str]:
        try:
            with open('stock_tickers.txt', 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            return []

    async def get_watchlist(self):
        """Get watchlist from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{PORTFOLIO_API_URL}/watchlist') as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []

    async def is_in_portfolio(self, ticker: str) -> bool:
        """Check if stock is in portfolio via API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{PORTFOLIO_API_URL}/positions/{ticker}') as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Error checking portfolio: {e}")
            return False

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
            
            # Add or update stocks in watchlist with fundamental scores
            async with aiohttp.ClientSession() as session:
                for stock in fund_results:
                    try:
                        # Check if stock exists in watchlist
                        check_response = await session.get(f'{PORTFOLIO_API_URL}/watchlist/{stock["ticker"]}')
                        
                        if check_response.status == 200:
                            # Update existing stock
                            logger.info(f"Updating {stock['ticker']} in watchlist")
                            await session.patch(
                                f'{PORTFOLIO_API_URL}/watchlist/{stock["ticker"]}',
                                json={
                                    'fundamental_score': stock['score'],
                                    'notes': f"Fundamental analysis updated: {datetime.now().isoformat()}"
                                }
                            )
                        else:
                            # Add new stock
                            logger.info(f"Adding {stock['ticker']} to watchlist")
                            await session.post(
                                f'{PORTFOLIO_API_URL}/watchlist',
                                json={
                                    'ticker': stock['ticker'],
                                    'fundamental_score': stock['score']
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error processing {stock['ticker']}: {e}")
                        continue
            
            logger.info(f"Updated watchlist with {len(fund_results)} stocks")
            return fund_results
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return []

    async def analyze_watchlist(self):
        """Analyze stocks in watchlist"""
        try:
            watchlist = await self.get_watchlist()
            
            for stock in watchlist:
                ticker = stock['ticker']
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        # Get technical analysis
                        tech_analysis = self.tech_analyzer.analyze_stock(ticker)
                        if not tech_analysis or 'technical_score' not in tech_analysis:
                            logger.error(f"Invalid technical analysis for {ticker}")
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.info(f"Retrying technical analysis for {ticker} (attempt {retry_count + 1})")
                                await asyncio.sleep(5)  # Wait 5 seconds before retry
                                continue
                            break
                        
                        # Get news analysis
                        news_analysis = self.news_analyzer.analyze_stock_news(ticker)
                        if not news_analysis or 'news_score' not in news_analysis:
                            logger.error(f"Invalid news analysis for {ticker}")
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.info(f"Retrying news analysis for {ticker} (attempt {retry_count + 1})")
                                await asyncio.sleep(5)
                                continue
                            break
                        
                        # Get current price
                        current_price = tech_analysis.get('signals', {}).get('current_price')
                        if current_price is None:
                            try:
                                stock_info = yf.Ticker(ticker).info
                                current_price = stock_info.get('regularMarketPrice', 0.0)
                            except Exception as e:
                                logger.error(f"Error getting current price for {ticker}: {e}")
                                current_price = 0.0
                        
                        # Log analysis results for debugging
                        logger.info(f"Analysis results for {ticker}:")
                        logger.info(f"Technical Score: {tech_analysis['technical_score']['total']:.2f}")
                        logger.info("Technical Scores by Timeframe:")
                        for tf, score in tech_analysis['technical_score']['timeframes'].items():
                            logger.info(f"  {tf}: {score:.2f}")
                        logger.info(f"News Score: {news_analysis['news_score']}")
                        logger.info(f"Current Price: {current_price}")
                        
                        # Prepare update data with proper type conversion
                        update_data = {
                            'last_analysis': datetime.now().isoformat(),
                            'technical_score': float(tech_analysis['technical_score']['total']),
                            'technical_scores': {
                                tf: float(score) 
                                for tf, score in tech_analysis['technical_score']['timeframes'].items()
                            },
                            'news_score': float(news_analysis['news_score']),
                            'news_sentiment': news_analysis['sentiment'],
                            'risk_level': tech_analysis['signals']['volatility']['risk_level'],
                            'current_price': float(current_price) if current_price else 0.0
                        }
                        
                        # Update watchlist item with retries
                        update_retry_count = 0
                        while update_retry_count < max_retries:
                            try:
                                await self.update_watchlist_item(ticker, update_data)
                                logger.info(f"Successfully updated watchlist item for {ticker}")
                                break
                            except Exception as e:
                                update_retry_count += 1
                                if update_retry_count < max_retries:
                                    logger.warning(f"Retry {update_retry_count} updating watchlist item for {ticker}: {e}")
                                    await asyncio.sleep(5)
                                else:
                                    logger.error(f"Failed to update watchlist item for {ticker} after {max_retries} attempts: {e}")
                        
                        # Check if we should create a position
                        if self.should_create_position(tech_analysis, news_analysis):
                            await self.handle_portfolio_addition(
                                ticker,
                                tech_analysis['technical_score']['total'],
                                tech_analysis,
                                news_analysis
                            )
                        
                        # Successfully processed this stock, break the retry loop
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Retry {retry_count} analyzing {ticker}: {e}")
                            await asyncio.sleep(5)
                        else:
                            logger.error(f"Failed to analyze {ticker} after {max_retries} attempts: {e}")
                            logger.error(f"Technical Analysis: {tech_analysis if 'tech_analysis' in locals() else 'Not available'}")
                            logger.error(f"News Analysis: {news_analysis if 'news_analysis' in locals() else 'Not available'}")
                
        except Exception as e:
            logger.error(f"Error in watchlist analysis: {e}")
            # Don't raise the exception to keep the loop running

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
        try:
            while True:
                current_time = datetime.now()
                
                # Run fundamental analysis every 24 hours
                if (self.last_fundamental_run is None or 
                    (current_time - self.last_fundamental_run).total_seconds() >= FUNDAMENTAL_ANALYSIS_INTERVAL):
                    logger.info("Running fundamental analysis...")
                    fund_results = await self.run_fundamental_analysis()
                    self.last_fundamental_run = current_time
                    
                    # Immediately run technical and news analysis on new stocks
                    if fund_results:
                        logger.info("Running initial analysis on new stocks...")
                        await self.analyze_watchlist()
                
                # Run technical analysis every 15 minutes
                await self.analyze_watchlist()
                
                # Run news analysis every 5 minutes
                watchlist = await self.get_watchlist()
                for stock in watchlist:
                    ticker = stock['ticker']
                    try:
                        news_analysis = self.news_analyzer.analyze_stock_news(ticker)
                        
                        # Update watchlist with news scores
                        update_data = {
                            'news_score': news_analysis['news_score'],
                            'news_sentiment': news_analysis['sentiment'],
                            'last_news': current_time.isoformat()
                        }
                        
                        await self.update_watchlist_item(ticker, update_data)
                        
                        # Check for significant news changes
                        old_score = float(stock.get('news_score', 50))
                        score_change = abs(news_analysis['news_score'] - old_score)
                        
                        if (score_change > 15 or  # Significant change in sentiment
                            news_analysis['news_score'] >= 70 or  # Very positive news
                            news_analysis['news_score'] <= 30):   # Very negative news
                            
                            logger.info(f"Significant news change for {ticker} (change: {score_change:.2f})")
                            await self.check_position_opportunity(ticker, news_analysis)
                        
                    except Exception as e:
                        logger.error(f"Error monitoring news for {ticker}: {e}")
                        continue
                
                # Wait for 5 minutes before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
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
            # Get technical signals
            signals = tech_analysis['signals']['indicators']
            support_resistance = tech_analysis['support_resistance']
            trend = tech_analysis['trend']
            current_price = tech_analysis['signals']['current_price']
            
            # Prepare position data for API
            position_data = {
                'ticker': ticker,
                'entry_price': float(tech_analysis['signals'].get('current_price', 0)),
                'current_price': current_price,
                'entry_date': datetime.now().isoformat(),
                'fundamental_score': score * 0.4,  # 40% of combined score
                'technical_score': tech_analysis['technical_score']['total_score'],
                'news_score': news_analysis['news_score'],
                'overall_score': score,
                'pnl': 0,  # Initial PnL
                'timeframe': self.tech_analyzer.timeframe,  # 'medium' or 'long'
                'status': 'open',
                'technical_data': {
                    'support_levels': support_resistance['support']['levels'][:3],
                    'resistance_levels': support_resistance['resistance']['levels'][:3],
                    'trend': {
                        'direction': trend['trend'],
                        'strength': trend['strength']
                    },
                    'signals': {
                        'rsi': signals['momentum']['rsi'],
                        'macd': {
                            'value': signals['trend']['macd'],
                            'signal': signals['trend']['macd_trend']
                        },
                        'predicted_move': tech_analysis['predictions']['final']['predicted_change_percent']
                    }
                }
            }
            
            # Send position to backend API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://portfolio-tracker-rough-dawn-5271.fly.dev/api/positions',
                    json=position_data
                ) as response:
                    if response.status != 201:
                        raise Exception(f"Failed to create position: {await response.text()}")
            
            # Send notification
            message = (
                f"🚨 <b>New Portfolio Addition</b> 🚨\n"
                f"Ticker: {ticker}\n"
                f"Entry Price: ${current_price:.2f}\n"
                f"Timeframe: {self.tech_analyzer.timeframe}\n"
                f"Scores:\n"
                f"- Technical: {tech_analysis['technical_score']['total_score']:.1f}\n"
                f"- Fundamental: {score * 0.4:.1f}\n"
                f"- News: {news_analysis['news_score']:.1f}\n"
                f"- Overall: {score:.1f}\n"
                f"Expected Move: {tech_analysis['predictions']['final']['predicted_change_percent']:.1f}%\n"
                f"Support Levels: ${', $'.join([f'{x:.2f}' for x in position_data['technical_data']['support_levels']])}\n"
                f"Resistance Levels: ${', $'.join([f'{x:.2f}' for x in position_data['technical_data']['resistance_levels']])}"
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
            # Get entry data from API
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{PORTFOLIO_API_URL}/positions/{ticker}') as response:
                    if response.status != 200:
                        raise Exception(f"Failed to get position data: {response.status}")
                    entry_data = await response.json()
            
            entry_price = entry_data['entry_price']
            current_price = tech_analysis['signals']['current_price']
            
            # Calculate return
            returns_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Remove from portfolio using API
            async with aiohttp.ClientSession() as session:
                async with session.delete(f'{PORTFOLIO_API_URL}/positions/{ticker}') as response:
                    if response.status not in (200, 204):
                        raise Exception(f"Failed to remove position: {response.status}")
            
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
        """Store analysis results using API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{PORTFOLIO_API_URL}/analysis',
                    json={
                        'ticker': ticker,
                        'timestamp': datetime.now().isoformat(),
                        'analysis_data': analysis
                    }
                ) as response:
                    if response.status not in (200, 201):
                        logger.error(f"Failed to store analysis: {response.status}")
        except Exception as e:
            logger.error(f"Error storing analysis for {ticker}: {e}")

    async def update_position_analysis(self, ticker: str, tech_analysis: Dict):
        """Update position with latest technical analysis"""
        try:
            # Get current position data
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://portfolio-tracker-rough-dawn-5271.fly.dev/api/positions/{ticker}'
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to get position: {await response.text()}")
                    position = await response.json()
            
            # Calculate new values
            current_price = tech_analysis['signals']['current_price']
            entry_price = position['entry_price']
            pnl = ((current_price - entry_price) / entry_price) * 100
            
            # Prepare update data
            update_data = {
                'current_price': current_price,
                'pnl': pnl,
                'technical_data': {
                    'support_levels': tech_analysis['support_resistance']['support']['levels'][:3],
                    'resistance_levels': tech_analysis['support_resistance']['resistance']['levels'][:3],
                    'trend': {
                        'direction': tech_analysis['trend']['trend'],
                        'strength': tech_analysis['trend']['strength']
                    },
                    'signals': {
                        'rsi': tech_analysis['signals']['indicators']['momentum']['rsi'],
                        'macd': {
                            'value': tech_analysis['signals']['indicators']['trend']['macd'],
                            'signal': tech_analysis['signals']['indicators']['trend']['macd_trend']
                        },
                        'predicted_move': tech_analysis['predictions']['final']['predicted_change_percent']
                    }
                }
            }
            
            # Update position in backend
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f'https://portfolio-tracker-rough-dawn-5271.fly.dev/api/positions/{ticker}',
                    json=update_data
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to update position: {await response.text()}")
            
            # Check for exit signals
            if self.should_exit_position(update_data):
                await self.handle_position_exit(ticker, update_data)
            
        except Exception as e:
            logger.error(f"Error updating position analysis for {ticker}: {e}")

    def should_exit_position(self, position_data: Dict) -> bool:
        """Determine if position should be exited based on technical signals"""
        # Exit if stop loss hit
        if position_data['current_price'] <= position_data['stop_loss']:
            return True
        
        # Exit if take profit hit
        if position_data['current_price'] >= position_data['take_profit']:
            return True
        
        # Exit on strong technical signals
        if (position_data['trend']['direction'] == 'strong_bearish' and 
            position_data['signals']['macd']['signal'] == 'sell' and
            position_data['trend']['strength'] > 70):
            return True
        
        return False

    async def handle_position_exit(self, ticker: str, position_data: Dict):
        """Handle position exit"""
        try:
            # Send exit notification
            message = (
                f"🔔 <b>Position Exit Signal</b> 🔔\n"
                f"Ticker: {ticker}\n"
                f"Exit Price: ${position_data['current_price']:.2f}\n"
                f"PnL: {position_data['pnl']:+.2f}%\n"
                f"Reason: {self.get_exit_reason(position_data)}\n"
                f"Technical Signals:\n"
                f"- Trend: {position_data['trend']['direction']}\n"
                f"- MACD: {position_data['signals']['macd']['signal']}\n"
                f"- Predicted Move: {position_data['signals']['predicted_move']:+.1f}%"
            )
            
            await self.send_telegram_alert(message)
            
            # Update position status to closed
            async with aiohttp.ClientSession() as session:
                await session.patch(
                    f'{PORTFOLIO_API_URL}/positions/{ticker}',
                    json={'status': 'closed'}
                )
            
        except Exception as e:
            logger.error(f"Error handling position exit for {ticker}: {e}")

    def get_exit_reason(self, position_data: Dict) -> str:
        """Determine the reason for position exit"""
        if position_data['current_price'] <= position_data.get('stop_loss', 0):
            return "Stop Loss Hit"
        
        if position_data['current_price'] >= position_data.get('take_profit', float('inf')):
            return "Take Profit Hit"
        
        if (position_data['trend']['direction'] == 'strong_bearish' and 
            position_data['signals']['macd']['signal'] == 'sell' and
            position_data['trend']['strength'] > 70):
            return "Strong Bearish Technical Signals"
        
        return "Multiple Technical Indicators"

    def should_create_position(self, tech_analysis: Dict, news_analysis: Dict) -> bool:
        """Check if we should create a position based on analysis"""
        try:
            logger.info("\n=== Position Criteria Check ===")
            
            # Get scores
            tech_score = tech_analysis['technical_score']['total']
            news_score = news_analysis['news_score']
            
            # Log all scores for debugging
            logger.info(f"Technical Score: {tech_score:.2f}")
            logger.info(f"Technical Scores by Timeframe:")
            for tf, score in tech_analysis['technical_score']['timeframes'].items():
                logger.info(f"  {tf}: {score:.2f}")
            logger.info(f"News Score: {news_score:.2f}")
            
            # Technical criteria - require strong alignment
            timeframe_scores = tech_analysis['technical_score']['timeframes']
            aligned_bullish = all(score >= 60 for score in timeframe_scores.values())
            
            # Updated news criteria - more lenient
            news_criteria = (
                news_score >= 40 and  # Lowered threshold
                news_analysis['sentiment'] in ['positive', 'neutral']  # Allow neutral
            )
            
            # Risk check
            risk_level = tech_analysis['signals']['volatility']['risk_level']
            risk_acceptable = risk_level in ['low', 'medium']
            
            # Log decision factors
            logger.info("\nDecision Factors:")
            logger.info(f"Technical Score: {tech_score:.2f}")
            logger.info(f"Timeframe Alignment - Bullish: {aligned_bullish}")
            logger.info(f"News Score ({news_score:.2f}) >= 40: {news_criteria}")
            logger.info(f"Risk Level ({risk_level}): {risk_acceptable}")
            
            # Final decision - require strong technical setup
            should_enter = (
                (aligned_bullish and tech_score >= 65) and  # Strong technical setup
                news_criteria and  # More lenient news requirement
                risk_acceptable
            )
            
            logger.info(f"\nFinal Decision: {'ENTER' if should_enter else 'SKIP'}")
            return should_enter
            
        except Exception as e:
            logger.error(f"Error checking position criteria: {e}")
            return False

    async def analyze_technical(self):
        """Run technical analysis on portfolio positions"""
        try:
            # Get current positions from portfolio
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{PORTFOLIO_API_URL}/positions') as response:
                    if response.status != 200:
                        logger.error("Failed to get positions")
                        return
                    positions = await response.json()

            for position in positions:
                ticker = position['ticker']
                try:
                    # Run technical analysis
                    analysis = self.tech_analyzer.analyze_stock(ticker)
                    
                    # Check exit signals and update position
                    exit_signal, updates = await self.check_exit_signals(ticker, position, analysis)
                    
                    # Update position with latest technical analysis data
                    technical_updates = {
                        'last_analysis': {
                            'timestamp': datetime.now().isoformat(),
                            'technical_score': analysis['summary']['highest_score'],
                            'trend': analysis['signals']['trend']['direction'],
                            'rsi': analysis['signals']['momentum']['rsi'],
                            'volume_profile': analysis['signals']['volume']['profile'],
                            'risk_level': analysis['signals']['volatility']['risk_level']
                        }
                    }

                    # If we already have updates, merge them
                    if updates:
                        technical_updates.update(updates)

                    # Update position with latest analysis
                    await self._update_position(ticker, technical_updates)

                    # If exit signal, close position
                    if exit_signal:
                        await self._close_position(ticker, {
                            'status': 'closed',
                            'exit_price': analysis['signals']['current_price'],
                            'exit_date': datetime.now().isoformat(),
                            'exit_reason': updates.get('exit_reason', 'technical_signal')
                        })

                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in technical analysis run: {e}")

    async def _close_position(self, ticker: str, close_data: Dict):
        """Close a position in the portfolio"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{PORTFOLIO_API_URL}/positions/{ticker}/close"
                async with session.patch(url, json=close_data) as response:
                    if response.status not in (200, 201):
                        logger.error(f"Error closing position for {ticker}: {response.status}")
                        return False
                    logger.info(f"Closed position for {ticker}: {close_data}")
                    return True
        except Exception as e:
            logger.error(f"API error closing position for {ticker}: {e}")
            return False

    async def monitor_news(self):
        """Continuously monitor news for watchlist stocks"""
        while True:
            try:
                watchlist = await self.get_watchlist()
                for stock in watchlist:
                    ticker = stock['ticker']
                    try:
                        news_analysis = self.news_analyzer.analyze_stock_news(ticker)
                        
                        # Update watchlist with news scores
                        async with aiohttp.ClientSession() as session:
                            await session.patch(
                                f'{PORTFOLIO_API_URL}/watchlist/{ticker}',
                                json={
                                    'news_score': news_analysis['news_score'],
                                    'last_news': datetime.now().isoformat()
                                }
                            )
                        
                        # If significant news, trigger technical analysis check
                        if news_analysis['news_score'] > 70:  # Significant positive news
                            logger.info(f"Significant news for {ticker}, running technical check")
                            await self.check_position_opportunity(ticker, news_analysis)
                            
                    except Exception as e:
                        logger.error(f"Error monitoring news for {ticker}: {e}")
                        continue
                        
                # Small delay between news checks - Changed from 30 to 45
                await asyncio.sleep(45)
                    
            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(45)  # Changed from 60 to 45

    async def test_integration(self):
        """Test integration between components"""
        try:
            # Test technical analysis
            tech_analyzer = TechnicalAnalyzer()
            tech_analysis = tech_analyzer.analyze_stock("AAPL")
            
            # Verify analysis results
            assert tech_analysis is not None
            assert 'signals' in tech_analysis
            assert 'technical_score' in tech_analysis
            
            logger.info("Integration test passed successfully")
            return True
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False

    async def check_exit_signals(self, ticker: str, position: Dict, analysis: Dict):
        """Check if position should be exited and manage targets"""
        current_price = analysis['signals']['current_price']
        entry_price = position['entry_price']
        risk_level = analysis['signals']['volatility']['risk_level']
        profit_pct = ((current_price - entry_price) / entry_price)
        
        # Get profit targets based on risk level
        targets = PROFIT_TARGETS[risk_level]
        position_updates = {}
        exit_signal = False
        
        # Check stop loss
        stop_loss = position.get('stop_loss') or analysis['support_resistance']['support'][0]
        if current_price <= stop_loss:
            await self._send_sell_signal_alert(
                ticker, analysis, "Stop Loss Hit", entry_price
            )
            return True, {'status': 'closed', 'exit_reason': 'stop_loss'}

        # Check and update profit targets
        if not position.get('first_target_hit') and profit_pct >= targets['first_target']:
            # First target hit - take partial profits
            await self._send_sell_signal_alert(
                ticker, analysis, 
                f"First Profit Target ({targets['first_target']*100}%) Hit - Taking Partial Profits", 
                entry_price
            )
            position_updates.update({
                'first_target_hit': True,
                'partial_exit': True,
                'partial_exit_price': current_price,
                'remaining_size': 0.5,  # Reduce position size by half
                # Update stop loss to entry price to secure profits
                'stop_loss': entry_price
            })

        if position.get('first_target_hit') and not position.get('final_target_hit') and profit_pct >= targets['final_target']:
            # Final target hit
            await self._send_sell_signal_alert(
                ticker, analysis, 
                f"Final Profit Target ({targets['final_target']*100}%) Hit", 
                entry_price
            )
            position_updates.update({
                'final_target_hit': True,
                'exit_price': current_price
            })
            exit_signal = True

        # Update trailing stop if activated
        if profit_pct >= TRAILING_STOP['activation']:
            current_trailing_stop = position.get('trailing_stop')
            new_trailing_stop = entry_price + (current_price * (1 - TRAILING_STOP['distance']))
            
            # Update trailing stop if it's higher than current
            if not current_trailing_stop or new_trailing_stop > current_trailing_stop:
                position_updates['trailing_stop'] = new_trailing_stop
                position_updates['highest_price'] = current_price
            
            # Check if price hit trailing stop
            if current_trailing_stop and current_price <= current_trailing_stop:
                await self._send_sell_signal_alert(
                    ticker, analysis, 
                    f"Trailing Stop Hit at {TRAILING_STOP['distance']*100}% from high", 
                    entry_price
                )
                exit_signal = True

        # Check technical conditions for reversal
        if (analysis['summary']['highest_score'] < 50 and 
            analysis['signals']['trend']['direction'] in ['bearish', 'strong_bearish']):
            # Only exit if we've hit at least first target, otherwise just update stop loss
            if position.get('first_target_hit'):
                await self._send_sell_signal_alert(
                    ticker, analysis, "Technical Reversal After First Target", entry_price
                )
                exit_signal = True
            else:
                # Update stop loss if technical conditions deteriorate
                new_stop = max(stop_loss, analysis['support_resistance']['support'][0])
                position_updates['stop_loss'] = new_stop

        # If we have updates, apply them
        if position_updates:
            await self._update_position(ticker, position_updates)
        
        return exit_signal, position_updates

    async def _update_position(self, ticker: str, updates: Dict):
        """Update position details in portfolio API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{PORTFOLIO_API_URL}/positions/{ticker}"
                async with session.patch(url, json=updates) as response:
                    if response.status not in (200, 201):
                        logger.error(f"Error updating position for {ticker}: {response.status}")
                        return False
                    logger.info(f"Updated position for {ticker}: {updates}")
                    return True
        except Exception as e:
            logger.error(f"API error updating position for {ticker}: {e}")
            return False

    async def update_watchlist_item(self, ticker: str, update_data: Dict) -> bool:
        """Update a watchlist item with new analysis data"""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if item exists
                check_url = f'{PORTFOLIO_API_URL}/watchlist/{ticker}'
                async with session.get(check_url) as response:
                    if response.status == 404:
                        # Item doesn't exist, create it
                        create_data = {
                            'ticker': ticker,
                            'technical_score': update_data.get('technical_score', 50),
                            'news_score': update_data.get('news_score', 50),
                            'news_sentiment': update_data.get('news_sentiment', 'neutral'),
                            'risk_level': update_data.get('risk_level', 'medium'),
                            'current_price': update_data.get('current_price', 0.0),
                            'last_updated': datetime.now().isoformat()
                        }
                        async with session.post(f'{PORTFOLIO_API_URL}/watchlist', json=create_data) as create_response:
                            if create_response.status not in (200, 201):
                                text = await create_response.text()
                                logger.error(f"Failed to create watchlist item for {ticker}. Status: {create_response.status}, Response: {text}")
                                return False
                            logger.info(f"Created new watchlist item for {ticker}")
                            return True

                # Item exists, update it
                async with session.patch(check_url, json=update_data) as update_response:
                    if update_response.status != 200:
                        text = await update_response.text()
                        logger.error(f"Failed to update watchlist item for {ticker}. Status: {update_response.status}, Response: {text}")
                        return False
                    logger.info(f"Updated watchlist item for {ticker}")
                    return True

        except Exception as e:
            logger.error(f"API error for {ticker}: {e}")
            return False

    async def analyze_single_stock(self, ticker: str):
        """Analyze a single stock immediately after passing fundamental criteria"""
        try:
            logger.info(f"Running immediate analysis for {ticker}")
            
            # Run technical analysis with retries
            retry_count = 0
            max_retries = 3
            tech_analysis = None
            
            while retry_count < max_retries:
                try:
                    tech_analysis = self.tech_analyzer.analyze_stock(ticker)
                    if tech_analysis and 'technical_score' in tech_analysis:
                        break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Retry {retry_count} technical analysis for {ticker}: {e}")
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"Failed technical analysis for {ticker} after {max_retries} attempts: {e}")
                        return
            
            # Run news analysis with retries
            retry_count = 0
            news_analysis = None
            
            while retry_count < max_retries:
                try:
                    news_analysis = self.news_analyzer.analyze_stock_news(ticker)
                    if news_analysis and 'news_score' in news_analysis:
                        break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Retry {retry_count} news analysis for {ticker}: {e}")
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"Failed news analysis for {ticker} after {max_retries} attempts: {e}")
                        return
            
            if tech_analysis and news_analysis:
                # Get current price
                current_price = tech_analysis.get('signals', {}).get('current_price')
                if current_price is None:
                    try:
                        stock_info = yf.Ticker(ticker).info
                        current_price = stock_info.get('regularMarketPrice', 0.0)
                    except Exception as e:
                        logger.error(f"Error getting current price for {ticker}: {e}")
                        current_price = 0.0
                
                # Prepare update data
                update_data = {
                    'last_analysis': datetime.now().isoformat(),
                    'technical_score': float(tech_analysis['technical_score']['total']),
                    'technical_scores': {
                        tf: float(score) 
                        for tf, score in tech_analysis['technical_score']['timeframes'].items()
                    },
                    'news_score': float(news_analysis['news_score']),
                    'news_sentiment': news_analysis['sentiment'],
                    'risk_level': tech_analysis['signals']['volatility']['risk_level'],
                    'current_price': float(current_price) if current_price else 0.0
                }
                
                # Update watchlist item
                await self.update_watchlist_item(ticker, update_data)
                
                # Check if we should create a position
                if self.should_create_position(tech_analysis, news_analysis):
                    await self.handle_portfolio_addition(
                        ticker,
                        tech_analysis['technical_score']['total'],
                        tech_analysis,
                        news_analysis
                    )
                
                logger.info(f"Completed immediate analysis for {ticker}")
            
        except Exception as e:
            logger.error(f"Error in immediate analysis for {ticker}: {e}")