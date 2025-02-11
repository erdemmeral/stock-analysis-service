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
    PORTFOLIO_API_URL
)

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
        """Run technical and news analysis on watchlist stocks"""
        try:
            # Get ALL stocks from watchlist instead of pending
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{PORTFOLIO_API_URL}/watchlist') as response:
                    if response.status != 200:
                        logger.error(f"Failed to get watchlist: {response.status}")
                        return
                    try:
                        watchlist = await response.json()
                        if not watchlist:
                            logger.info("No stocks in watchlist")
                            return
                        
                        logger.info(f"Retrieved {len(watchlist)} stocks for analysis")
                        # Log the actual tickers for debugging
                        logger.info(f"Stocks to analyze: {[stock['ticker'] for stock in watchlist]}")
                    except Exception as e:
                        text = await response.text()
                        logger.error(f"Failed to parse watchlist response: {text}")
                        return

            # Analyze each stock in watchlist
            for stock in watchlist:
                ticker = stock['ticker']
                try:
                    logger.info(f"\n=== Starting analysis for {ticker} ===")
                    
                    # Run analyses
                    tech_analysis = self.tech_analyzer.analyze_stock(ticker)
                    logger.info(f"{ticker} technical analysis complete")
                    
                    news_analysis = self.news_analyzer.analyze_stock_news(ticker)
                    logger.info(f"{ticker} news analysis complete")
                    
                    # Log analysis results
                    logger.info(f"{ticker} Analysis Results:")
                    logger.info(f"Current Price: ${tech_analysis['signals'].get('current_price', 'N/A')}")
                    logger.info(f"Technical Score: {tech_analysis['technical_score']['total_score']}")
                    logger.info(f"News Score: {news_analysis['news_score']}")
                    
                    async with aiohttp.ClientSession() as session:
                        # Update watchlist with scores
                        await session.patch(
                            f'{PORTFOLIO_API_URL}/watchlist/{ticker}',
                            json={
                                'technical_score': tech_analysis['technical_score']['total_score'],
                                'news_score': news_analysis['news_score'],
                                'last_analysis': datetime.now().isoformat(),
                                'notes': f"Last analyzed: {datetime.now().isoformat()}"
                            }
                        )
                        logger.info(f"Updated {ticker} in watchlist")

                        # Check position criteria
                        if self.should_create_position(tech_analysis, news_analysis):
                            logger.info(f"✅ {ticker} meets position criteria, creating position")
                            position_data = {
                                "ticker": ticker,
                                "entry_price": float(tech_analysis['signals'].get('current_price', 0)),
                                "timeframe": "medium",
                                "technical_score": tech_analysis['technical_score']['total_score'],
                                "news_score": news_analysis['news_score'],
                                "support_levels": tech_analysis['support_resistance']['support']['levels'][:3],
                                "resistance_levels": tech_analysis['support_resistance']['resistance']['levels'][:3],
                                "trend": {
                                    "direction": tech_analysis['trend']['trend'],
                                    "strength": tech_analysis['trend']['strength'],
                                    "ma_alignment": tech_analysis['signals']['moving_averages']['ma_aligned']
                                },
                                "signals": {
                                    "rsi": tech_analysis['signals']['momentum']['rsi'],
                                    "macd": {
                                        "value": tech_analysis['signals']['trend']['macd'],
                                        "signal": tech_analysis['signals']['trend']['macd_trend']
                                    },
                                    "volume_profile": tech_analysis['signals']['volume']['profile'],
                                    "predicted_move": tech_analysis['predictions']['final']['predicted_change_percent']
                                }
                            }
                            
                            position_response = await session.post(
                                f'{PORTFOLIO_API_URL}/positions',
                                json=position_data
                            )
                            if position_response.status == 200:
                                logger.info(f"Successfully created position for {ticker}")
                            else:
                                logger.error(f"Failed to create position for {ticker}: {position_response.status}")
                        else:
                            logger.info(f"❌ {ticker} did not meet position criteria")
                    
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
                await asyncio.sleep(self.analysis_interval)
                
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
        try:
            logger.info(f"\n=== Position Criteria Check ===")
            
            # 1. Fundamental Score Check (must be in watchlist)
            # Stock must have passed fundamental analysis to be in watchlist
            
            # 2. Technical Analysis Checks
            tech_score = tech_analysis['technical_score']['total_score']
            trend = tech_analysis['trend']
            signals = tech_analysis['signals']
            
            logger.info(f"Technical Score: {tech_score}")
            logger.info(f"Trend Direction: {trend['direction']}")
            logger.info(f"Trend Strength: {trend['strength']}")
            logger.info(f"RSI: {signals['momentum']['rsi']}")
            logger.info(f"MACD: {signals['trend']['macd_trend']}")
            logger.info(f"Volume: {signals['volume']['profile']}")

            # Dynamic Technical Score threshold based on other factors
            min_tech_score = 75  # Base threshold
            if trend['direction'] == 'strong_bullish' and news_analysis['news_score'] > 70:
                min_tech_score = 65  # Lower threshold for strong trends with great news
            
            if tech_score < min_tech_score:
                logger.info(f"❌ Failed: Technical score {tech_score} below threshold {min_tech_score}")
                return False

            # Trend must be bullish
            if trend['direction'] not in ['bullish', 'strong_bullish']:
                logger.info("❌ Failed: Not in bullish trend")
                return False

            # Dynamic RSI thresholds based on trend strength
            rsi = signals['momentum']['rsi']
            rsi_max = 75 if trend['direction'] == 'strong_bullish' else 70
            if rsi > rsi_max:
                logger.info(f"❌ Failed: RSI {rsi} above threshold {rsi_max}")
                return False

            # MACD confirmation
            if signals['trend']['macd_trend'] != 'bullish':
                # Allow non-bullish MACD only if all other signals are very strong
                if not (trend['direction'] == 'strong_bullish' and 
                       tech_score > 80 and 
                       news_analysis['news_score'] > 65):
                    logger.info("❌ Failed: MACD not bullish and other signals not strong enough")
                    return False

            # Volume requirement based on price action
            if signals['volume']['profile'] not in ['increasing', 'high']:
                logger.info("❌ Failed: Volume not supportive")
                return False

            # 3. News Analysis Checks
            news_score = news_analysis['news_score']
            logger.info(f"News Score: {news_score}")
            
            # Dynamic news threshold based on technical strength
            min_news_score = 55 if tech_score > 85 else 60
            if news_score < min_news_score:
                logger.info(f"❌ Failed: News score {news_score} below threshold {min_news_score}")
                return False

            # 4. Support/Resistance Check
            current_price = float(tech_analysis['signals']['current_price'])
            resistance = tech_analysis['support_resistance']['resistance']['levels'][0]
            support = tech_analysis['support_resistance']['support']['levels'][0]
            
            # Allow breakout scenarios
            price_range = resistance - support
            breakout_threshold = resistance + (price_range * 0.02)  # 2% above resistance
            
            if support < current_price < resistance:
                # Price between support and resistance
                price_to_resistance = resistance - current_price
                price_to_support = current_price - support
                if price_to_support > price_to_resistance:
                    logger.info("❌ Failed: Price closer to resistance than support")
                    return False
            elif current_price > breakout_threshold:
                # Confirmed breakout
                logger.info("✅ Breakout scenario detected")
            else:
                logger.info("❌ Failed: Price not in optimal position")
                return False

            logger.info("✅ All criteria passed! Creating position...")
            return True

        except Exception as e:
            logger.error(f"Error checking position criteria: {e}")
            return False

    async def analyze_technical(self):
        """Run technical analysis on watchlist stocks"""
        try:
            # Get watchlist
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{PORTFOLIO_API_URL}/watchlist') as response:
                    if response.status != 200:
                        logger.error(f"Failed to get watchlist: {response.status}")
                        return
                    watchlist = await response.json()
                    if not watchlist:
                        logger.info("No stocks in watchlist")
                        return
                    
                    logger.info(f"Running technical analysis on {len(watchlist)} stocks")
                    logger.info(f"Stocks: {[stock['ticker'] for stock in watchlist]}")

            for stock in watchlist:
                ticker = stock['ticker']
                try:
                    tech_analysis = self.tech_analyzer.analyze_stock(ticker)
                    logger.info(f"Technical analysis complete for {ticker}")
                    
                    # Update watchlist with technical scores
                    async with aiohttp.ClientSession() as session:
                        await session.patch(
                            f'{PORTFOLIO_API_URL}/watchlist/{ticker}',
                            json={
                                'technical_score': tech_analysis['technical_score']['total_score'],
                                'last_technical': datetime.now().isoformat()
                            }
                        )
                    
                    # Check position criteria with latest news
                    news_analysis = await self.get_latest_news_analysis(ticker)
                    if self.should_create_position(tech_analysis, news_analysis):
                        await self.create_position(ticker, tech_analysis, news_analysis)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")

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
                        
                # Small delay between news checks
                await asyncio.sleep(30)
                    
            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(60) 