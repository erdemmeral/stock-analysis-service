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
                        position_decision = self.should_create_position(tech_analysis, news_analysis)
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

    async def handle_portfolio_addition(self, ticker: str, score: float, tech_analysis: Dict, news_analysis: Dict):
        """Handle adding a position to the portfolio"""
        try:
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

            # Prepare position data with all required fields
            position_data = {
                'ticker': ticker,
                'entry_price': float(current_price),
                'timeframe': best_timeframe,
                'stop_loss': float(stop_loss),
                'technical_score': float(timeframe_scores[best_timeframe]),
                'news_score': float(news_analysis.get('news_score', 50)),
                'support_levels': tech_analysis.get('support_resistance', {}).get('support', []),
                'resistance_levels': tech_analysis.get('support_resistance', {}).get('resistance', []),
                'trend': tech_analysis.get('signals', {}).get('trend', {}).get('direction', 'neutral'),
                'risk_level': risk_level,
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

    def should_create_position(self, tech_analysis: Dict, news_analysis: Dict) -> Dict:
        """Determine if we should create a position based on technical and news analysis"""
        try:
            reasons = []
            # Get technical scores for all timeframes
            technical_scores = tech_analysis.get('technical_score', {}).get('timeframes', {})
            if not technical_scores:
                return {'create': False, 'reasons': ['No technical scores available']}

            # Find best timeframe score
            best_score = max(technical_scores.values())
            best_timeframe = max(technical_scores.items(), key=lambda x: x[1])[0]
            
            # Get news score
            news_score = news_analysis.get('news_score', 0)
            
            # Check volume profile
            volume_profile = tech_analysis.get('signals', {}).get('volume', {}).get('profile', 'low')
            has_good_volume = volume_profile in ['high', 'increasing', 'normal']
            
            # Check volatility
            volatility = tech_analysis.get('signals', {}).get('volatility', {})
            risk_level = volatility.get('risk_level', 'high')
            volatility_trend = volatility.get('trend', 'increasing')
            
            # Check each criterion and add failure reasons
            if best_score < 60:
                reasons.append(f'Technical score too low: {best_score:.1f} < 60')
            
            if news_score < 40:
                reasons.append(f'News sentiment too negative: {news_score:.1f} < 40')
            
            if not has_good_volume and best_timeframe not in ['medium', 'long']:
                reasons.append(f'Insufficient volume ({volume_profile}) for {best_timeframe} timeframe')
            
            if risk_level == 'high':
                fundamental_score = tech_analysis.get('fundamental_score', 0)
                if fundamental_score <= 80:
                    reasons.append(f'High risk ({risk_level}) with insufficient fundamental score: {fundamental_score:.1f} ≤ 80')
            elif risk_level == 'extreme':
                reasons.append(f'Extreme risk level: {risk_level}')
            
            # Additional checks for trend alignment
            trend_direction = tech_analysis.get('signals', {}).get('trend', {}).get('direction', 'neutral')
            if trend_direction in ['strong_bearish', 'bearish']:
                reasons.append(f'Bearish trend detected: {trend_direction}')
            
            # Check momentum
            rsi = tech_analysis.get('signals', {}).get('momentum', {}).get('rsi', 50)
            if rsi > 70:
                reasons.append(f'RSI overbought: {rsi:.1f} > 70')
            elif rsi < 30:
                reasons.append(f'RSI oversold: {rsi:.1f} < 30')
            
            # Log decision details
            if not reasons:
                logger.info(f"Position creation criteria met: tech_score={best_score}, "
                          f"news_score={news_score}, volume={volume_profile}, "
                          f"risk={risk_level}, timeframe={best_timeframe}")
                return {'create': True, 'reasons': ['All criteria met']}
            else:
                logger.info(f"Position creation criteria not met. Reasons: {', '.join(reasons)}")
                return {'create': False, 'reasons': reasons}
            
        except Exception as e:
            logger.error(f"Error in should_create_position: {e}")
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
        """Update watchlist item with new analysis"""
        try:
            # Get existing watchlist item data
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{PORTFOLIO_API_URL}/watchlist/{ticker}") as response:
                    if response.status == 200:
                        existing_data = await response.json()
                    else:
                        existing_data = {}

            # Ensure we have the technical scores - preserve existing if not in update
            technical_scores = update_data.get('technical_scores', existing_data.get('technical_scores', {}))
            if not technical_scores and 'technical_score' in update_data:
                logger.info(f"Converting single technical score to timeframe scores for {ticker}")
                technical_scores = {
                    'medium': float(update_data['technical_score'])
                }
            elif not technical_scores and 'technical_scores' in existing_data:
                technical_scores = existing_data['technical_scores']
                logger.info(f"Preserving existing technical scores for {ticker}")

            # Get best timeframe based on highest score
            if technical_scores:
                # Sort by score and get the highest scoring timeframe
                sorted_timeframes = sorted(technical_scores.items(), key=lambda x: float(x[1]), reverse=True)
                best_timeframe = sorted_timeframes[0][0]
                best_score = float(sorted_timeframes[0][1])
                
                # Log timeframe scores for debugging
                logger.info(f"Technical scores for {ticker}:")
                for tf, score in sorted_timeframes:
                    logger.info(f"{tf}: {score:.2f}")
            else:
                logger.warning(f"No technical scores available for {ticker}")
                best_timeframe = existing_data.get('best_timeframe', 'medium')
                best_score = existing_data.get('technical_score', 0.0)

            # Get current price with multiple fallbacks
            current_price = update_data.get('current_price')
            if current_price is None or current_price == 0:
                try:
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('regularMarketPrice')
                    if not current_price:
                        hist = stock.history(period='1d')
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                except Exception as e:
                    logger.error(f"Error fetching current price for {ticker}: {e}")
                    current_price = 0.0

            # Prepare watchlist data - preserve existing values if not in update
            watchlist_data = {
                'ticker': ticker,
                'last_analysis': datetime.now().isoformat(),
                'technical_score': best_score,
                'technical_scores': technical_scores,
                'news_score': float(update_data.get('news_score', existing_data.get('news_score', 50))),
                'news_sentiment': update_data.get('news_sentiment', existing_data.get('news_sentiment', 'neutral')),
                'risk_level': update_data.get('risk_level', existing_data.get('risk_level', 'medium')),
                'current_price': float(current_price) if current_price else float(existing_data.get('current_price', 0.0)),
                'best_timeframe': best_timeframe
            }

            # Log the update for debugging
            logger.info(f"Updating watchlist for {ticker}:")
            logger.info(f"Best Timeframe: {best_timeframe}")
            logger.info(f"Technical Score: {best_score:.2f}")
            logger.info(f"Current Price: {current_price}")

            # Try to update first
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.patch(
                        f"{PORTFOLIO_API_URL}/watchlist/{ticker}",
                        json=watchlist_data
                    ) as response:
                        if response.status == 404:
                            # If item doesn't exist, create it
                            async with session.post(
                                f"{PORTFOLIO_API_URL}/watchlist",
                                json=watchlist_data
                            ) as create_response:
                                if create_response.status not in (200, 201):
                                    error_text = await create_response.text()
                                    logger.error(f"Failed to create watchlist item for {ticker}. Status: {create_response.status}, Error: {error_text}")
                                    return False
                                logger.info(f"Created new watchlist item for {ticker}")
                        elif response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Failed to update watchlist item for {ticker}. Status: {response.status}, Error: {error_text}")
                            return False
                        else:
                            logger.info(f"Successfully updated watchlist item for {ticker}")
                except Exception as e:
                    logger.error(f"API call error for {ticker}: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error updating watchlist item for {ticker}: {e}")
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
                position_decision = self.should_create_position(tech_analysis, news_analysis)
                if position_decision['create']:
                    await self.handle_portfolio_addition(
                        ticker,
                        tech_analysis['technical_score']['total'],
                        tech_analysis,
                        news_analysis
                    )
                
                logger.info(f"Completed immediate analysis for {ticker}")
            
        except Exception as e:
            logger.error(f"Error in immediate analysis for {ticker}: {e}")

    async def add_to_watchlist(self, ticker: str, watchlist_data: Dict) -> bool:
        """Add a stock to the watchlist"""
        try:
            # Ensure required fields are present
            if 'ticker' not in watchlist_data:
                watchlist_data['ticker'] = ticker
            
            if 'fundamental_score' not in watchlist_data:
                logger.error(f"Missing fundamental_score for {ticker}")
                return False
            
            # Add additional required fields if missing
            if 'last_updated' not in watchlist_data:
                watchlist_data['last_updated'] = datetime.now().isoformat()
            
            if 'status' not in watchlist_data:
                watchlist_data['status'] = 'new'

            async with aiohttp.ClientSession() as session:
                # Check if already exists
                check_url = f'{PORTFOLIO_API_URL}/watchlist/{ticker}'
                async with session.get(check_url) as response:
                    exists = response.status == 200
                
                if exists:
                    # Update existing entry
                    async with session.patch(check_url, json=watchlist_data) as update_response:
                        if update_response.status != 200:
                            error_text = await update_response.text()
                            logger.error(f"Failed to update watchlist entry for {ticker}. Error: {error_text}")
                            return False
                        logger.info(f"Updated existing watchlist entry for {ticker}")
                else:
                    # Create new entry
                    create_url = f'{PORTFOLIO_API_URL}/watchlist'
                    async with session.post(create_url, json=watchlist_data) as create_response:
                        if create_response.status not in (200, 201):
                            error_text = await create_response.text()
                            logger.error(f"Failed to create watchlist entry for {ticker}. Error: {error_text}")
                            return False
                        logger.info(f"Created new watchlist entry for {ticker}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding {ticker} to watchlist: {e}")
            return False