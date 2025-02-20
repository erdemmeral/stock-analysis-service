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
import time

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        self.fund_analyzer = FundamentalAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        # Initialize bot for version 13.x
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
            # For python-telegram-bot v13.x
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
                        
                        watchlist_data = {
                            'ticker': stock['ticker'],
                            'fundamental_score': float(stock['score']),  # Ensure it's a float
                            'last_analysis': datetime.now().isoformat(),
                            'status': 'active'
                        }
                        
                        if check_response.status == 200:
                            # Update existing stock
                            logger.info(f"Updating {stock['ticker']} in watchlist with score {stock['score']}")
                            await session.patch(
                                f'{PORTFOLIO_API_URL}/watchlist/{stock["ticker"]}',
                                json=watchlist_data
                            )
                        else:
                            # Add new stock
                            logger.info(f"Adding {stock['ticker']} to watchlist with score {stock['score']}")
                            await session.post(
                                f'{PORTFOLIO_API_URL}/watchlist',
                                json=watchlist_data
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
                        
                        # Replace existing technical scores instead of adding new ones
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
                            'current_price': float(current_price) if current_price else 0.0,
                            'replace_scores': True  # Flag to indicate we want to replace existing scores
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
                        position_decision = await self.should_create_position(tech_analysis, news_analysis)
                        if position_decision['create']:
                            await self.handle_portfolio_addition(
                                ticker,
                                tech_analysis['technical_score']['total'],
                                tech_analysis,
                                news_analysis
                            )
                        else:
                            logger.info(f"Not creating position for {ticker}. Reasons: {', '.join(position_decision['reasons'])}")
                        
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
            cleanup_interval = 3600  # Run cleanup every hour
            last_cleanup = None
            
            while True:
                current_time = datetime.now()
                
                # Run cleanup every hour
                if last_cleanup is None or (current_time - last_cleanup).total_seconds() >= cleanup_interval:
                    logger.info("Running position cleanup...")
                    await self.clean_up_positions()
                    await self.clean_up_duplicate_positions()
                    last_cleanup = current_time
                
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

    async def handle_portfolio_addition(self, ticker: str, score: float, tech_analysis: Dict, news_analysis: Dict):
        """Handle adding a position to the portfolio"""
        try:
            # First check if stock is in watchlist
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{PORTFOLIO_API_URL}/watchlist/{ticker}") as response:
                    if response.status != 200:
                        logger.error(f"Cannot create position for {ticker}: Stock not in watchlist")
                        return False
                    watchlist_data = await response.json()
                    
            # Verify fundamental score exists
            if not watchlist_data.get('fundamental_score'):
                logger.error(f"Cannot create position for {ticker}: No fundamental score in watchlist")
                return False

            # Determine best timeframe based on scores
            timeframe_scores = tech_analysis.get('technical_score', {}).get('timeframes', {})
            if not timeframe_scores:
                logger.error(f"No timeframe scores available for {ticker}")
                return False
                
            best_timeframe = max(timeframe_scores.items(), key=lambda x: float(x[1]))[0]
            
            # Get current price
            current_price = tech_analysis.get('signals', {}).get('current_price')
            if not current_price:
                try:
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('regularMarketPrice', 0.0)
                except Exception as e:
                    logger.error(f"Error getting current price for {ticker}: {e}")
                    return False

            # Calculate stop loss based on support levels and volatility
            support_levels = tech_analysis.get('support_resistance', {}).get('support', [])
            resistance_levels = tech_analysis.get('support_resistance', {}).get('resistance', [])
            volatility = tech_analysis.get('signals', {}).get('volatility', {})
            risk_level = volatility.get('risk_level', 'medium')
            
            # Calculate stop loss using nearest support level or percentage based on risk
            if support_levels and len(support_levels) > 0:
                nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
                stop_loss = nearest_support
            else:
                # Default stop loss percentages based on risk level
                stop_loss_pcts = {'low': 0.05, 'medium': 0.07, 'high': 0.10}
                stop_pct = stop_loss_pcts.get(risk_level, 0.07)
                stop_loss = current_price * (1 - stop_pct)

            # Get trend information
            trend_data = tech_analysis.get('signals', {}).get('trend', {})
            trend_direction = trend_data.get('direction', 'neutral')
            trend_strength = trend_data.get('strength', 50)
            
            # Get momentum indicators
            momentum_data = tech_analysis.get('signals', {}).get('momentum', {})
            rsi = momentum_data.get('rsi', 50)
            macd_data = tech_analysis.get('signals', {}).get('indicators', {}).get('trend', {})
            
            # Get volume information
            volume_data = tech_analysis.get('signals', {}).get('volume', {})
            
            # Calculate alignment score based on trend, momentum, and volume
            alignment_factors = {
                'trend_aligned': trend_direction in ['bullish', 'strong_bullish'],
                'momentum_aligned': 40 <= rsi <= 60,
                'volume_aligned': volume_data.get('profile') in ['high', 'increasing']
            }
            alignment_score = sum(1 for aligned in alignment_factors.values() if aligned) * 100 / 3

            # Prepare position data with all required fields
            position_data = {
                'ticker': ticker,
                'entry_price': float(current_price),
                'timeframe': best_timeframe,
                'technical_scores': {
                    tf: float(score) for tf, score in timeframe_scores.items()
                },
                'news_score': float(news_analysis.get('news_score', 50)),
                'support_levels': support_levels[:3],  # Top 3 support levels
                'resistance_levels': resistance_levels[:3],  # Top 3 resistance levels
                'trend': {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'alignment_score': alignment_score
                },
                'signals': {
                    'rsi': float(rsi),
                    'macd': {
                        'value': float(macd_data.get('macd', 0)),
                        'signal': macd_data.get('signal', 'neutral')
                    },
                    'volume_profile': volume_data.get('profile', 'normal'),
                    'predicted_move': float(tech_analysis.get('predictions', {}).get('predicted_change_percent', 0))
                },
                'risk_level': risk_level,
                'stop_loss': float(stop_loss),
                'status': 'active',
                'created_at': datetime.now().isoformat()
            }

            # Create position via API
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f'{PORTFOLIO_API_URL}/positions',
                        json=position_data
                    ) as response:
                        if response.status not in (200, 201):
                            error_text = await response.text()
                            logger.error(f"Failed to create position for {ticker}. Status: {response.status}, Error: {error_text}")
                            return False
                        
                        logger.info(f"Successfully created position for {ticker}")
                        return True
                except Exception as e:
                    logger.error(f"API call error creating position for {ticker}: {e}")
                    return False

        except Exception as e:
            logger.error(f"Error creating position for {ticker}: {e}")
            return False

    async def handle_portfolio_removal(self, ticker: str, score: float, tech_analysis: Dict, news_analysis: Dict):
        """Handle removing a position from the portfolio"""
        try:
            current_price = tech_analysis['signals']['current_price']
            
            # Send sell signal via API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{PORTFOLIO_API_URL}/positions/{ticker}/sell',
                    json={
                        'soldPrice': current_price,
                        'soldDate': datetime.now().isoformat(),
                        'sellCondition': 'signal'
                    }
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to process sell signal for {ticker}")
                        return False
                    
                    logger.info(f"Successfully processed sell signal for {ticker}")
                    return True

        except Exception as e:
            logger.error(f"Error processing sell signal for {ticker}: {e}")
            return False

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

    async def check_position_opportunity(self, ticker: str, news_analysis: Dict):
        """Check if news creates a position opportunity"""
        try:
            # Run technical analysis to verify opportunity
            tech_analysis = self.tech_analyzer.analyze_stock(ticker)
            if not tech_analysis:
                logger.warning(f"Could not get technical analysis for {ticker}")
                return

            # Add ticker to tech_analysis for should_create_position
            tech_analysis['ticker'] = ticker
            
            # Check if we should create a position
            position_decision = await self.should_create_position(tech_analysis, news_analysis)
            if position_decision['create']:
                await self.handle_portfolio_addition(
                    ticker,
                    tech_analysis['technical_score']['total'],
                    tech_analysis,
                    news_analysis
                )
            else:
                logger.info(f"News opportunity for {ticker} did not meet position criteria: {position_decision['reasons']}")
                
        except Exception as e:
            logger.error(f"Error checking position opportunity for {ticker}: {e}")

    async def should_create_position(self, tech_analysis: Dict, news_analysis: Dict) -> Dict:
        """Determine if we should create a position based on technical and news analysis"""
        try:
            # Get ticker from tech_analysis
            ticker = tech_analysis.get('ticker')
            if not ticker:
                return {'create': False, 'reasons': ['No ticker provided in analysis']}
            
            # Check if position already exists - More robust check
            try:
                async with aiohttp.ClientSession() as session:
                    headers = self._get_api_headers()
                    # Check all positions regardless of status
                    positions_url = self._get_api_url('v1/positions')  # Updated endpoint
                    logger.info(f"Checking positions at URL: {positions_url}")
                    
                    async with session.get(positions_url, headers=headers) as response:
                        if response.status == 200:
                            try:
                                positions = await response.json()
                                # Check for any position with this ticker
                                for position in positions:
                                    if position.get('ticker') == ticker:
                                        status = position.get('status', 'unknown')
                                        logger.info(f"Position exists for {ticker} with status: {status}")
                                        return {'create': False, 'reasons': [f'Position already exists with status: {status}']}
                            except Exception as e:
                                error_text = await response.text()
                                logger.error(f"Error parsing positions response for {ticker}: {e}, Response: {error_text}, URL: {positions_url}")
                                return {'create': False, 'reasons': [f'Error checking positions: {str(e)}']}
                        else:
                            error_text = await response.text()
                            logger.error(f"Error checking positions. Status: {response.status}, Response: {error_text}, URL: {positions_url}")
                            return {'create': False, 'reasons': [f'Error checking positions. Status: {response.status}']}
            except Exception as e:
                logger.error(f"Error checking existing positions for {ticker}: {e}")
                return {'create': False, 'reasons': [f'Error checking positions: {str(e)}']}
            
            # Get technical scores for all timeframes
            technical_scores = tech_analysis.get('technical_score', {}).get('timeframes', {})
            if not technical_scores:
                return {'create': False, 'reasons': ['No technical scores available']}

            # Find best timeframe score, filtering out None values
            valid_scores = {k: float(v) for k, v in technical_scores.items() if v is not None}
            if not valid_scores:
                return {'create': False, 'reasons': ['No valid technical scores available']}
                
            best_score = max(valid_scores.values())
            best_timeframe = max(valid_scores.items(), key=lambda x: x[1])[0]
            
            # Log all scores for debugging
            logger.info(f"Technical scores for {ticker}: {technical_scores}")
            logger.info(f"Best score for {ticker}: {best_score} ({best_timeframe})")
            
            # Get news score, ensure it's a float
            news_score = float(news_analysis.get('news_score', 0))
            logger.info(f"News score for {ticker}: {news_score}")
            
            # Check volume profile
            volume_profile = tech_analysis.get('signals', {}).get('volume', {}).get('profile', 'low')
            has_good_volume = volume_profile in ['high', 'increasing', 'normal']
            logger.info(f"Volume profile for {ticker}: {volume_profile}")
            
            # Check volatility
            volatility = tech_analysis.get('signals', {}).get('volatility', {})
            risk_level = volatility.get('risk_level', 'high')
            volatility_trend = volatility.get('trend', 'increasing')
            logger.info(f"Risk level for {ticker}: {risk_level}, Volatility trend: {volatility_trend}")
            
            # Check each criterion and add failure reasons
            reasons = []
            
            if best_score < 55:  # Lowered from 60
                reasons.append(f'Technical score too low: {best_score:.1f} < 55')
            
            if news_score < 35:  # Lowered from 40
                reasons.append(f'News sentiment too negative: {news_score:.1f} < 35')
            
            if not has_good_volume:
                reasons.append(f'Insufficient volume ({volume_profile})')
            
            # Get fundamental score from watchlist
            try:
                async with aiohttp.ClientSession() as session:
                    headers = self._get_api_headers()
                    async with session.get(f'{PORTFOLIO_API_URL}/watchlist/{ticker}', headers=headers) as response:
                        if response.status == 200:
                            watchlist_data = await response.json()
                            fundamental_score = float(watchlist_data.get('fundamental_score', 0))
                            logger.info(f"Retrieved fundamental score from watchlist for {ticker}: {fundamental_score}")
                        else:
                            fundamental_score = 0
                            logger.warning(f"Could not get fundamental score for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching fundamental score from watchlist for {ticker}: {e}")
                fundamental_score = 0
            
            # Check fundamental score requirements based on risk level
            if risk_level == 'high':
                if fundamental_score < 70:  # Lowered from 80
                    reasons.append(f'High risk ({risk_level}) with insufficient fundamental score: {fundamental_score:.1f} < 70')
            elif risk_level == 'extreme':
                reasons.append(f'Extreme risk level: {risk_level}')
            
            # Additional checks for trend alignment
            trend_direction = tech_analysis.get('signals', {}).get('trend', {}).get('direction', 'neutral')
            logger.info(f"Trend direction for {ticker}: {trend_direction}")
            if trend_direction == 'strong_bearish':  # Only block strong bearish
                reasons.append(f'Strong bearish trend detected: {trend_direction}')
            
            # Check momentum
            momentum = tech_analysis.get('signals', {}).get('momentum', {})
            rsi = float(momentum.get('rsi', 50)) if momentum else 50
            logger.info(f"RSI for {ticker}: {rsi}")
            if rsi > 75:  # Increased from 70
                reasons.append(f'RSI overbought: {rsi:.1f} > 75')
            elif rsi < 25:  # Decreased from 30
                reasons.append(f'RSI oversold: {rsi:.1f} < 25')
            
            # Check if we have enough timeframe data
            required_timeframes = ['short']  # Only require short timeframe
            missing_timeframes = [tf for tf in required_timeframes if tf not in valid_scores]
            if missing_timeframes:
                reasons.append(f'Missing required timeframe data: {", ".join(missing_timeframes)}')
            
            # Log decision details
            if not reasons:
                logger.info(f"Position creation criteria met for {ticker}: tech_score={best_score}, "
                          f"news_score={news_score}, volume={volume_profile}, "
                          f"risk={risk_level}, timeframe={best_timeframe}, "
                          f"fundamental_score={fundamental_score}")
                return {'create': True, 'reasons': ['All criteria met']}
            else:
                logger.info(f"Position creation criteria not met for {ticker}. Reasons: {', '.join(reasons)}")
                return {'create': False, 'reasons': reasons}
            
        except Exception as e:
            logger.error(f"Error in should_create_position for {ticker}: {e}")
            return {'create': False, 'reasons': [f'Error: {str(e)}']}

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
        """Close a position"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_api_headers()
                url = self._get_api_url(f'v1/positions/{ticker}')
                async with session.patch(url, json=close_data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to close position for {ticker}. Status: {response.status}, Response: {error_text}")
                        return False
                    logger.info(f"Closed position for {ticker}")
                    return True
        except Exception as e:
            logger.error(f"Error closing position for {ticker}: {e}")
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
                headers = self._get_api_headers()
                url = self._get_api_url(f'v1/positions/{ticker}')
                async with session.patch(url, json=updates, headers=headers) as response:
                    if response.status not in (200, 201):
                        error_text = await response.text()
                        logger.error(f"Error updating position for {ticker}: Status {response.status}, Response: {error_text}")
                        return False
                    logger.info(f"Updated position for {ticker}: {updates}")
                    return True
        except Exception as e:
            logger.error(f"API error updating position for {ticker}: {e}")
            return False

    async def update_watchlist_item(self, ticker: str, update_data: Dict) -> bool:
        """Update a watchlist item with new analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_api_headers()
                
                # First check if item exists
                watchlist_url = self._get_api_url(f'v1/watchlist/{ticker}')  # Updated endpoint
                async with session.get(watchlist_url, headers=headers) as response:
                    if response.status == 200:
                        # Update existing item
                        async with session.patch(
                            watchlist_url,
                            json=update_data,
                            headers=headers
                        ) as update_response:
                            if update_response.status != 200:
                                error_text = await update_response.text()
                                logger.error(f"Failed to update watchlist item for {ticker}. Status: {update_response.status}, Response: {error_text}")
                                return False
                            logger.info(f"Updated watchlist item for {ticker}")
                            return True
                    elif response.status == 404:
                        # Create new item
                        create_url = self._get_api_url('v1/watchlist')  # Updated endpoint
                        async with session.post(
                            create_url,
                            json={"ticker": ticker, **update_data},
                            headers=headers
                        ) as create_response:
                            if create_response.status not in (200, 201):
                                error_text = await create_response.text()
                                logger.error(f"Failed to create watchlist item for {ticker}. Status: {create_response.status}, Response: {error_text}")
                                return False
                            logger.info(f"Created new watchlist item for {ticker}")
                            return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Error checking watchlist for {ticker}. Status: {response.status}, Response: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error updating watchlist item for {ticker}: {e}")
            return False

    async def analyze_single_stock(self, ticker: str):
        """Analyze a single stock and update watchlist"""
        try:
            # Get technical analysis
            tech_analysis = self.tech_analyzer.analyze_stock(ticker)
            
            # Get news analysis
            news_analysis = self.news_analyzer.analyze_stock_news(ticker)
            
            # Get current price from technical analysis
            current_price = tech_analysis.get('signals', {}).get('current_price', 0.0)
            
            # Prepare watchlist update data
            update_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat(),
                'technical_score': float(tech_analysis['technical_score']['total']),
                'technical_scores': tech_analysis['technical_score']['timeframes'],
                'news_score': float(news_analysis['news_score']),
                'news_sentiment': news_analysis['sentiment'],
                'risk_level': tech_analysis.get('signals', {}).get('volatility', {}).get('risk_level', 'medium'),
                'current_price': float(current_price)
            }
            
            # Log the data being sent
            logger.info(f"Updating watchlist for {ticker} with data: {update_data}")
            
            # Update watchlist
            await self.update_watchlist_item(ticker, update_data)
            
            return {
                'technical_analysis': tech_analysis,
                'news_analysis': news_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    async def add_to_watchlist(self, ticker: str, watchlist_data: Dict) -> bool:
        """Add or update a stock in the watchlist"""
        try:
            # Check if ticker exists in watchlist
            async with aiohttp.ClientSession() as session:
                headers = self._get_api_headers()
                async with session.get(f"{PORTFOLIO_API_URL}/watchlist/{ticker}", headers=headers) as response:
                    if response.status == 200:
                        # Update existing watchlist item
                        async with session.patch(
                            f"{PORTFOLIO_API_URL}/watchlist/{ticker}",
                            json=watchlist_data,
                            headers=headers
                        ) as update_response:
                            if update_response.status != 200:
                                error_text = await update_response.text()
                                logger.error(f"Failed to update watchlist item for {ticker}. Status: {update_response.status}, Response: {error_text}")
                                return False
                            logger.info(f"Updated watchlist item for {ticker}")
                            return True
                    elif response.status == 404:
                        # Create new watchlist item
                        async with session.post(
                            f"{PORTFOLIO_API_URL}/watchlist",
                            json={"ticker": ticker, **watchlist_data},
                            headers=headers
                        ) as create_response:
                            if create_response.status not in (200, 201):
                                error_text = await create_response.text()
                                logger.error(f"Failed to create watchlist item for {ticker}. Status: {create_response.status}, Response: {error_text}")
                                return False
                            logger.info(f"Created new watchlist item for {ticker}")
                            return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Error checking watchlist for {ticker}. Status: {response.status}, Response: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error adding {ticker} to watchlist: {e}")
            return False

    def _get_api_headers(self):
        """Get standard headers for API calls"""
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'StockAnalysisService/1.0'
        }

    def _get_api_url(self, endpoint: str) -> str:
        """Get full API URL for given endpoint"""
        # Ensure the base URL doesn't end with a slash
        base_url = PORTFOLIO_API_URL.rstrip('/')
        # Ensure the endpoint starts with a slash
        endpoint = f"/{endpoint.lstrip('/')}"
        return f"{base_url}{endpoint}"

    async def clean_up_positions(self):
        """Clean up positions that are not in watchlist"""
        try:
            # Get all active positions
            async with aiohttp.ClientSession() as session:
                headers = self._get_api_headers()
                positions_url = self._get_api_url('v1/positions')
                async with session.get(positions_url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to get positions for cleanup. Status: {response.status}, Response: {error_text}")
                        return
                    try:
                        positions = await response.json()
                    except Exception as e:
                        error_text = await response.text()
                        logger.error(f"Error parsing positions response: {e}, Response: {error_text}")
                        return
                
                # Get watchlist
                watchlist_url = self._get_api_url('v1/watchlist')
                async with session.get(watchlist_url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to get watchlist for cleanup. Status: {response.status}, Response: {error_text}")
                        return
                    try:
                        watchlist = await response.json()
                    except Exception as e:
                        error_text = await response.text()
                        logger.error(f"Error parsing watchlist response: {e}, Response: {error_text}")
                        return
                    
                # Create set of watchlist tickers for faster lookup
                watchlist_tickers = {item['ticker'] for item in watchlist}
                
                # Check each position
                for position in positions:
                    ticker = position['ticker']
                    if ticker not in watchlist_tickers:
                        logger.warning(f"Found position for {ticker} but not in watchlist. Closing position.")
                        # Close the position
                        await self._close_position(ticker, {
                            'status': 'closed',
                            'exit_reason': 'watchlist_cleanup',
                            'exit_date': datetime.now().isoformat()
                        })
                        
        except Exception as e:
            logger.error(f"Error in position cleanup: {e}")

    async def clean_up_duplicate_positions(self):
        """Clean up any duplicate positions for the same ticker"""
        try:
            # Get all positions
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{PORTFOLIO_API_URL}/positions") as response:
                    if response.status != 200:
                        logger.error("Failed to get positions for duplicate cleanup")
                        return
                    positions = await response.json()
                
                # Track positions by ticker
                ticker_positions = {}
                for position in positions:
                    ticker = position.get('ticker')
                    if not ticker:
                        continue
                        
                    if ticker not in ticker_positions:
                        ticker_positions[ticker] = []
                    ticker_positions[ticker].append(position)
                
                # Check for duplicates
                for ticker, pos_list in ticker_positions.items():
                    if len(pos_list) > 1:
                        logger.warning(f"Found {len(pos_list)} positions for {ticker}")
                        
                        # Sort by creation date to keep the oldest one
                        sorted_positions = sorted(pos_list, 
                                               key=lambda x: x.get('creation_date', ''),
                                               reverse=True)
                        
                        # Keep the oldest position, close others
                        for position in sorted_positions[1:]:
                            logger.info(f"Closing duplicate position for {ticker}: {position.get('_id')}")
                            await self._close_position(ticker, {
                                'status': 'closed',
                                'exit_reason': 'duplicate_cleanup',
                                'exit_date': datetime.now().isoformat()
                            })
                            
        except Exception as e:
            logger.error(f"Error in duplicate position cleanup: {e}")