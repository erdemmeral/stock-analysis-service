import logging
from typing import Dict, List
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
import yfinance as yf
from scipy.signal import argrelextrema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self, timeframe: str = 'medium'):
        self.timeframe = timeframe
        self.setup_indicators()
        logger.info(f"Initialized TechnicalAnalyzer with {timeframe} timeframe")

    def analyze_stock(self, ticker: str) -> Dict:
        """Analyze a stock and return technical analysis results"""
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y', interval='1d')
            
            if hist.empty:
                logger.error(f"No data available for {ticker}")
                return self.get_empty_analysis()
            
            # Get current price and add to signals
            current_price = float(hist['Close'].iloc[-1])
            logger.info(f"{ticker} current price: ${current_price:.2f}")
            
            # Calculate all analysis components
            signals = {
                'current_price': current_price,  # Add current price here
                'momentum': self.analyze_momentum(hist),
                'trend': self.analyze_trend_signals(hist),
                'volume': self.analyze_volume(hist),
                'moving_averages': self.analyze_moving_averages(hist)
            }
            
            trend = self.analyze_trend(hist)
            support_resistance = self.calculate_support_resistance(hist)
            technical_score = self.calculate_technical_score(signals, trend)
            
            return {
                'signals': signals,
                'trend': trend,
                'support_resistance': support_resistance,
                'technical_score': technical_score,
                'predictions': self.make_predictions(hist, signals, trend)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return self.get_empty_analysis()

    def setup_indicators(self):
        """
        Medium term: 3-6 months outlook
        Long term: 6-12 months outlook
        """
        self.indicators = {
            'medium': {
                'ma_short': 20,     # ~1 month
                'ma_long': 50,      # ~2.5 months
                'rsi_period': 14,   # Standard RSI
                'macd_fast': 12,    # Standard MACD
                'macd_slow': 26,
                'macd_signal': 9,
                'adx_period': 14,
                'fibonacci_period': 126  # ~6 months
            },
            'long': {
                'ma_short': 50,     # ~2.5 months
                'ma_long': 200,     # ~10 months
                'rsi_period': 14,   # Keep standard for reliability
                'macd_fast': 26,    # Slower MACD for long term
                'macd_slow': 52,    # ~2.5 months
                'macd_signal': 18,
                'adx_period': 21,
                'fibonacci_period': 252  # ~1 year
            }
        }
    
    def analyze_stocks_from_fundamental(self, fundamental_file: str) -> Dict:
        """
        Analyzes stocks that passed fundamental analysis
        """
        # Load fundamental analysis results
        with open(fundamental_file, 'r') as f:
            fund_results = json.load(f)
        
        passing_stocks = fund_results['passing_results']
        tech_results = []
        
        for stock in passing_stocks:
            ticker = stock['ticker']
            tech_analysis = self.analyze_stock(ticker)
            tech_results.append({
                'ticker': ticker,
                'fundamental_score': stock['score'],
                'technical_analysis': tech_analysis
            })
        
        return tech_results

    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculates technical indicators
        """
        try:
            params = self.indicators[self.timeframe]
            current_price = float(data['Close'].iloc[-1].item())
            
            # Moving Averages
            ma_short = float(ta.trend.sma_indicator(data['Close'], params['ma_short']).iloc[-1].item())
            ma_long = float(ta.trend.sma_indicator(data['Close'], params['ma_long']).iloc[-1].item())
            
            # Momentum Indicators
            rsi = float(ta.momentum.rsi(data['Close'], params['rsi_period']).iloc[-1].item())
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
            stoch_k = float(stoch.stoch().iloc[-1].item())
            stoch_d = float(stoch.stoch_signal().iloc[-1].item())
            williams_r = float(ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r().iloc[-1].item())
            
            # Trend Indicators
            macd = ta.trend.MACD(data['Close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
            macd_line = float(macd.macd().iloc[-1].item())
            signal_line = float(macd.macd_signal().iloc[-1].item())
            
            # Volume Indicators
            obv = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            mfi = float(ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume']).iloc[-1].item())

            return {
                'current_price': current_price,
                'indicators': {
                    'moving_averages': {
                        'ma_short': ma_short,
                        'ma_long': ma_long,
                        'ma_trend': 'bullish' if ma_short > ma_long else 'bearish'
                    },
                    'momentum': {
                        'rsi': rsi,
                        'rsi_signal': self._interpret_rsi(rsi),
                        'stochastic': stoch_k,
                        'stoch_signal': stoch_d,
                        'williams_r': williams_r
                    },
                    'trend': {
                        'macd': macd_line,
                        'macd_signal': signal_line,
                        'macd_trend': 'bullish' if macd_line > signal_line else 'bearish',
                        'trend_strength': self._calculate_trend_strength(data)
                    },
                    'volume': {
                        'obv': float(obv.iloc[-1].item()),
                        'obv_trend': self._calculate_trend(obv),
                        'mfi': mfi,
                        'mfi_signal': self._interpret_mfi(mfi)
                    }
                }
            }
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {
                'current_price': 0,
                'indicators': {
                    'moving_averages': {'ma_trend': 'unknown'},
                    'momentum': {'rsi_signal': 'unknown'},
                    'trend': {'macd_trend': 'unknown'},
                    'volume': {'obv_trend': 'unknown'}
                }
            }
    
    def get_technical_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetches historical price data and calculates technical indicators
        """
        # TODO: Implement API calls to get price data
        pass
    
    def score_technicals(self, technical_data: pd.DataFrame) -> float:
        """
        Scores the stock based on technical indicators
        Returns a score between 0 and 100
        """
        # TODO: Implement scoring logic
        pass 

    def find_support_resistance(self, data: pd.DataFrame, num_levels: int = 3) -> Dict:
        """
        Identifies support and resistance levels using multiple methods
        Adjusts window sizes based on timeframe
        """
        try:
            # Set different windows based on timeframe
            if self.timeframe == 'medium':
                local_window = 20    # ~1 month
                ma_windows = [20, 50]  # 20 and 50 day MAs
                volume_period = 126   # ~6 months
            else:  # long term
                local_window = 50    # ~2.5 months
                ma_windows = [50, 200]  # 50 and 200 day MAs
                volume_period = 252   # ~1 year
            
            current_price = data['Close'].iloc[-1]
            
            # Method 1: Find local minima and maxima using appropriate window
            highs = argrelextrema(data['High'].values, np.greater_equal, order=local_window)[0]
            lows = argrelextrema(data['Low'].values, np.less_equal, order=local_window)[0]
            
            # Get price levels
            resistance_levels = sorted(set(data['High'].iloc[highs]))
            support_levels = sorted(set(data['Low'].iloc[lows]))
            
            # Method 2: Moving Average Bounces with timeframe-specific MAs
            for window in ma_windows:
                ma = ta.trend.sma_indicator(data['Close'], window=window)
                if not pd.isna(ma.iloc[-1]):
                    resistance_levels.append(ma.iloc[-1])
            
            # Method 3: Volume Profile with timeframe-specific period
            if not data['Volume'].empty:
                recent_data = data.tail(volume_period)
                volume_profile = pd.DataFrame({
                    'price': recent_data['Close'],
                    'volume': recent_data['Volume']
                })
                volume_levels = volume_profile.groupby('price')['volume'].sum()
                high_volume_levels = volume_levels.nlargest(5).index.values
                
                resistance_levels.extend(high_volume_levels[high_volume_levels > current_price])
                support_levels.extend(high_volume_levels[high_volume_levels < current_price])
            
            # Filter and sort levels
            resistance_levels = sorted([level for level in set(resistance_levels) 
                                     if level > current_price and not pd.isna(level)])
            support_levels = sorted([level for level in set(support_levels) 
                                   if level < current_price and not pd.isna(level)], 
                                  reverse=True)
            
            # Get nearest levels
            nearest_resistance = resistance_levels[:num_levels] if resistance_levels else []
            nearest_support = support_levels[:num_levels] if support_levels else []
            
            # Calculate strength of levels with timeframe-specific parameters
            resistance_strength = self._calculate_level_strength(
                data, 
                nearest_resistance, 
                volume_period=volume_period
            )
            support_strength = self._calculate_level_strength(
                data, 
                nearest_support,
                volume_period=volume_period
            )
            
            return {
                'current_price': current_price,
                'resistance': {
                    'levels': nearest_resistance,
                    'strength': resistance_strength
                },
                'support': {
                    'levels': nearest_support,
                    'strength': support_strength
                }
            }
            
        except Exception as e:
            print(f"Error finding support/resistance levels: {str(e)}")
            return {
                'current_price': data['Close'].iloc[-1],
                'resistance': {'levels': [], 'strength': []},
                'support': {'levels': [], 'strength': []}
            }

    def _calculate_level_strength(self, data: pd.DataFrame, levels: list, volume_period: int = 126) -> list:
        """
        Calculates the strength of support/resistance levels based on timeframe
        """
        strengths = []
        recent_data = data.tail(volume_period)  # Use timeframe-specific period
        
        for level in levels:
            try:
                # Find touches (price coming within 0.5% of level)
                touches_mask = abs(recent_data['Close'] - level) / level < 0.005
                num_touches = sum(touches_mask)
                
                if num_touches == 0:
                    strengths.append(0)
                    continue
                
                # Get volume at level
                volume_at_level = recent_data.loc[touches_mask, 'Volume'].mean()
                volume_ratio = volume_at_level / recent_data['Volume'].mean()
                
                # Find the most recent touch
                touch_dates = recent_data.index[touches_mask]
                if len(touch_dates) > 0:
                    last_touch = touch_dates[-1]
                    days_since_touch = (recent_data.index[-1] - last_touch).days
                    # Adjust recency calculation based on timeframe
                    max_days = 180 if self.timeframe == 'medium' else 365
                    recency = max(0, 1 - (days_since_touch / max_days))
                else:
                    recency = 0
                
                # Combined strength score (0-100)
                touch_score = min(100, num_touches * 10)
                volume_score = min(100, volume_ratio * 30)
                recency_score = recency * 60
                
                strength = (touch_score + volume_score + recency_score) / 3
                strengths.append(min(100, strength))
                
            except Exception as e:
                print(f"Error calculating strength for level {level}: {str(e)}")
                strengths.append(0)
        
        return strengths

    def calculate_fibonacci_levels(self, data: pd.DataFrame) -> Dict:
        """
        Calculates Fibonacci retracement levels
        """
        period = self.indicators[self.timeframe]['fibonacci_period']
        recent_high = data['High'].tail(period).max()
        recent_low = data['Low'].tail(period).min()
        diff = recent_high - recent_low
        
        return {
            'level_0': recent_low,
            'level_0.236': recent_low + 0.236 * diff,
            'level_0.382': recent_low + 0.382 * diff,
            'level_0.5': recent_low + 0.5 * diff,
            'level_0.618': recent_low + 0.618 * diff,
            'level_1': recent_high
        }

    def analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze price trend using multiple methods"""
        try:
            # Get last values and convert to float
            ma_short = float(ta.trend.sma_indicator(data['Close'], 20).iloc[-1])
            ma_long = float(ta.trend.sma_indicator(data['Close'], 50).iloc[-1])
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate slopes (convert to float)
            short_slope = float(ma_short - ta.trend.sma_indicator(data['Close'], 20).iloc[-5])
            long_slope = float(ma_long - ta.trend.sma_indicator(data['Close'], 50).iloc[-5])
            
            # Get ADX for trend strength (convert to float)
            adx = float(ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx().iloc[-1])
            
            # Determine trend direction
            if current_price > ma_short > ma_long and short_slope > 0 and long_slope > 0:
                trend = 'strong_uptrend'
            elif current_price > ma_short > ma_long:
                trend = 'uptrend'
            elif current_price < ma_short < ma_long and short_slope < 0 and long_slope < 0:
                trend = 'strong_downtrend'
            elif current_price < ma_short < ma_long:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Calculate trend strength (0-100)
            strength = min(100, max(0, float(adx)))  # Ensure it's between 0-100
            
            return {
                'trend': trend,
                'strength': strength,
                'price_above_ma': bool(current_price > ma_short),
                'ma_alignment': bool(ma_short > ma_long),
                'momentum': 'increasing' if short_slope > 0 else 'decreasing'
            }
            
        except Exception as e:
            print(f"Error analyzing trend: {str(e)}")
            return {
                'trend': 'unknown',
                'strength': 0,
                'price_above_ma': False,
                'ma_alignment': False,
                'momentum': 'unknown'
            }

    def calculate_technical_score(self, data: pd.DataFrame, signals: Dict) -> Dict:
        """Calculate comprehensive technical score"""
        try:
            # Define weights for different components
            weights = {
                'trend': 0.30,
                'momentum': 0.25,
                'support_resistance': 0.15,
                'volatility': 0.15,
                'volume': 0.15
            }
            
            # 1. Trend Score (0-100)
            trend_signals = signals['indicators']
            trend_score = (
                (100 if trend_signals['moving_averages']['ma_trend'] == 'bullish' else 0) +
                (100 if trend_signals['trend']['macd_trend'] == 'bullish' else 0)
            ) / 2
            
            # 2. Momentum Score (0-100)
            momentum = trend_signals['momentum']
            momentum_score = (
                (100 if momentum['rsi_signal'] in ['oversold', 'bullish'] else 
                 0 if momentum['rsi_signal'] in ['overbought', 'bearish'] else 50) +
                (100 if momentum['stochastic'] > 80 else 
                 0 if momentum['stochastic'] < 20 else 50)
            ) / 2
            
            # 3. Volume Score (0-100)
            volume = trend_signals['volume']
            volume_score = (
                (100 if volume['obv_trend'] == 'up' else 
                 0 if volume['obv_trend'] == 'down' else 50) +
                (100 if volume['mfi'] > 80 else 
                 0 if volume['mfi'] < 20 else 50)
            ) / 2
            
            # 4. Volatility Score (0-100)
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            volatility_score = (
                100 if volatility < 0.15 else
                75 if volatility < 0.25 else
                50 if volatility < 0.35 else
                25
            )
            
            # 5. Support/Resistance Score (0-100)
            sr_levels = self.find_support_resistance(data)
            current_price = data['Close'].iloc[-1]
            
            if sr_levels['support']['levels'] and sr_levels['resistance']['levels']:
                nearest_support = sr_levels['support']['levels'][0]
                nearest_resistance = sr_levels['resistance']['levels'][0]
                
                # Calculate distance to nearest levels
                support_distance = (current_price - nearest_support) / current_price
                resistance_distance = (nearest_resistance - current_price) / current_price
                
                sr_score = (
                    100 if resistance_distance > support_distance else
                    0 if support_distance > resistance_distance else
                    50
                )
            else:
                sr_score = 50
            
            # Calculate component scores
            scores = {
                'trend_score': float(trend_score),
                'momentum_score': float(momentum_score),
                'support_resistance_score': float(sr_score),
                'volatility_score': float(volatility_score),
                'volume_score': float(volume_score)
            }
            
            # Calculate final weighted score
            final_score = sum(scores[f"{k}_score"] * v for k, v in weights.items())
            
            # Create interpretation
            interpretation = {
                'overall_rating': (
                    'very bullish' if final_score >= 80 else
                    'bullish' if final_score >= 60 else
                    'neutral' if final_score >= 40 else
                    'bearish' if final_score >= 20 else
                    'very bearish'
                ),
                'score': float(final_score),
                'summary': f"Technical analysis indicates {'bullish' if final_score > 50 else 'bearish'} outlook",
                'confidence': 'high' if abs(max(scores.values()) - min(scores.values())) < 30 else 'medium'
            }
            
            return {
                'total_score': float(final_score),
                'component_scores': scores,
                'weights': weights,
                'interpretation': interpretation
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return {
                'total_score': 50.0,
                'component_scores': {},
                'weights': {},
                'interpretation': {
                    'overall_rating': 'neutral',
                    'score': 50.0,
                    'summary': 'Error calculating score',
                    'confidence': 'low'
                }
            }

    def _calculate_trend_score(self, signals: Dict) -> float:
        """Calculate trend score from signals"""
        try:
            ma_trend = signals.get('moving_averages', {}).get('ma_trend', 'neutral')
            trend_score = 75 if ma_trend == 'bullish' else 25 if ma_trend == 'bearish' else 50
            return trend_score
        except Exception as e:
            logger.error(f"Error calculating trend score: {str(e)}")
            return 50

    def _calculate_momentum_score(self, signals: Dict) -> float:
        """Calculate momentum score"""
        try:
            rsi = signals.get('momentum', {}).get('rsi', 50)
            macd = signals.get('trend', {}).get('macd', 0)
            
            rsi_score = 100 if 40 <= rsi <= 60 else 50
            macd_score = 75 if macd > 0 else 25 if macd < 0 else 50
            
            return (rsi_score + macd_score) / 2
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return 50

    def _calculate_sr_score(self, data: pd.DataFrame) -> float:
        """Calculate support/resistance score"""
        try:
            sr_levels = self.find_support_resistance(data)
            nearest_support = sr_levels['support']['levels'][0] if sr_levels['support']['levels'] else 0
            nearest_resist = sr_levels['resistance']['levels'][0] if sr_levels['resistance']['levels'] else float('inf')
            current_price = sr_levels['current_price']
            
            sr_score = (
                (sr_levels['support']['strength'][0] if sr_levels['support']['strength'] else 0) +
                (sr_levels['resistance']['strength'][0] if sr_levels['resistance']['strength'] else 0)
            ) / 2
            
            return sr_score
        except Exception as e:
            logger.error(f"Error calculating sr_score: {str(e)}")
            return 0

    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility score"""
        try:
            # Calculate volatility metrics
            returns = data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # ATR calculation
            atr = ta.volatility.AverageTrueRange(
                data['High'], data['Low'], data['Close']
            ).average_true_range().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            bb_width = bb.bollinger_wband().iloc[-1]
            
            # Score components
            volatility_score = 100 * (1 - min(1, volatility))  # Lower volatility is better
            atr_score = 100 * (1 - min(1, atr / data['Close'].iloc[-1]))
            bb_score = 100 * (1 - min(1, bb_width))
            
            # Combined score
            return float((volatility_score + atr_score + bb_score) / 3)
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 50

    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume score"""
        try:
            # Recent volume vs average volume
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].tail(20).mean()
            volume_ratio = recent_volume / avg_volume
            
            # Volume trend
            volume_trend = 1 if recent_volume > avg_volume else -1
            
            # Price-volume correlation
            returns = data['Close'].pct_change()
            volume_changes = data['Volume'].pct_change()
            correlation = returns.corr(volume_changes)
            
            # Score components
            volume_trend_score = 75 if volume_trend > 0 else 25
            correlation_score = min(100, max(0, (correlation + 1) * 50))  # Scale -1 to 1 into 0-100
            consistency_score = min(100, max(0, volume_ratio * 50))
            
            # Combined score
            return float((volume_trend_score + correlation_score + consistency_score) / 3)
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 50

    def _calculate_pattern_score(self, data: pd.DataFrame) -> float:
        """Calculate pattern score"""
        try:
            patterns = self._check_patterns(data)
            
            # Score based on pattern reliability and confirmation
            pattern_score = 0
            if patterns['double_bottom'] or patterns['inverse_head_shoulders']:
                pattern_score = 100  # Strong bullish patterns
            elif patterns['double_top'] or patterns['head_shoulders']:
                pattern_score = 0    # Strong bearish patterns
            elif patterns['bullish_flag'] or patterns['ascending_triangle']:
                pattern_score = 75   # Moderate bullish patterns
            elif patterns['bearish_flag'] or patterns['descending_triangle']:
                pattern_score = 25   # Moderate bearish patterns
            else:
                pattern_score = 50   # No clear patterns
            
            return float(pattern_score)
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {str(e)}")
            return 50

    def _check_patterns(self, data: pd.DataFrame) -> Dict:
        """Check for various chart patterns"""
        try:
            patterns = {
                'double_top': self._check_double_top(data),
                'double_bottom': self._check_double_bottom(data),
                'head_shoulders': self._check_head_shoulders(data),
                'inverse_head_shoulders': self._check_inverse_head_shoulders(data),
                'bullish_flag': self._check_bullish_flag(data),
                'bearish_flag': self._check_bearish_flag(data),
                'ascending_triangle': self._check_ascending_triangle(data),
                'descending_triangle': self._check_descending_triangle(data)
            }
            return patterns
        except Exception as e:
            logger.error(f"Error checking patterns: {str(e)}")
            return dict.fromkeys([
                'double_top', 'double_bottom', 'head_shoulders', 
                'inverse_head_shoulders', 'bullish_flag', 'bearish_flag',
                'ascending_triangle', 'descending_triangle'
            ], False)

    def _check_bullish_flag(self, data: pd.DataFrame) -> bool:
        """Check for bullish flag pattern"""
        try:
            # Get recent price data
            closes = data['Close'].iloc[-20:]  # Last 20 periods
            highs = data['High'].iloc[-20:]
            lows = data['Low'].iloc[-20:]
            
            # Check for prior uptrend (pole)
            uptrend = closes.iloc[-15:-5].is_monotonic_increasing
            
            # Check for consolidation (flag)
            recent_highs = highs.iloc[-5:]
            recent_lows = lows.iloc[-5:]
            
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # Flag should be slightly downward or sideways
            is_flag = (-0.01 <= high_slope <= 0.005) and (-0.01 <= low_slope <= 0.005)
            
            return uptrend and is_flag
        except Exception as e:
            logger.error(f"Error checking bullish flag: {str(e)}")
            return False

    def _check_bearish_flag(self, data: pd.DataFrame) -> bool:
        """Check for bearish flag pattern"""
        try:
            # Get recent price data
            closes = data['Close'].iloc[-20:]
            highs = data['High'].iloc[-20:]
            lows = data['Low'].iloc[-20:]
            
            # Check for prior downtrend (pole)
            downtrend = closes.iloc[-15:-5].is_monotonic_decreasing
            
            # Check for consolidation (flag)
            recent_highs = highs.iloc[-5:]
            recent_lows = lows.iloc[-5:]
            
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # Flag should be slightly upward or sideways
            is_flag = (-0.005 <= high_slope <= 0.01) and (-0.005 <= low_slope <= 0.01)
            
            return downtrend and is_flag
        except Exception as e:
            logger.error(f"Error checking bearish flag: {str(e)}")
            return False

    def _check_ascending_triangle(self, data: pd.DataFrame) -> bool:
        """Check for ascending triangle pattern"""
        try:
            # Get recent price data
            highs = data['High'].iloc[-20:]
            lows = data['Low'].iloc[-20:]
            
            # Find resistance line (horizontal)
            resistance = highs.max()
            resistance_touches = sum(abs(highs - resistance) / resistance < 0.01)
            
            # Find support line (ascending)
            lows_array = np.array(range(len(lows)))
            low_slope = np.polyfit(lows_array, lows, 1)[0]
            
            # Check criteria
            has_resistance = resistance_touches >= 3
            has_ascending_support = low_slope > 0
            
            return has_resistance and has_ascending_support
        except Exception as e:
            logger.error(f"Error checking ascending triangle: {str(e)}")
            return False

    def _check_descending_triangle(self, data: pd.DataFrame) -> bool:
        """Check for descending triangle pattern"""
        try:
            # Get recent price data
            highs = data['High'].iloc[-20:]
            lows = data['Low'].iloc[-20:]
            
            # Find support line (horizontal)
            support = lows.min()
            support_touches = sum(abs(lows - support) / support < 0.01)
            
            # Find resistance line (descending)
            highs_array = np.array(range(len(highs)))
            high_slope = np.polyfit(highs_array, highs, 1)[0]
            
            # Check criteria
            has_support = support_touches >= 3
            has_descending_resistance = high_slope < 0
            
            return has_support and has_descending_resistance
        except Exception as e:
            logger.error(f"Error checking descending triangle: {str(e)}")
            return False

    def _check_double_top(self, data: pd.DataFrame) -> bool:
        """Check for double top pattern"""
        try:
            highs = data['High'].rolling(window=5).max()
            # Look for two similar highs with a trough in between
            peaks = argrelextrema(highs.values, np.greater_equal, order=5)[0]
            if len(peaks) >= 2:
                peak1, peak2 = peaks[-2:]
                if abs(highs.iloc[peak1] - highs.iloc[peak2]) / highs.iloc[peak1] < 0.02:  # Within 2%
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking double top: {str(e)}")
            return False

    def _check_double_bottom(self, data: pd.DataFrame) -> bool:
        """Check for double bottom pattern"""
        try:
            lows = data['Low'].rolling(window=5).min()
            # Look for two similar lows with a peak in between
            troughs = argrelextrema(lows.values, np.less_equal, order=5)[0]
            if len(troughs) >= 2:
                trough1, trough2 = troughs[-2:]
                if abs(lows.iloc[trough1] - lows.iloc[trough2]) / lows.iloc[trough1] < 0.02:  # Within 2%
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking double bottom: {str(e)}")
            return False

    def _check_head_shoulders(self, data: pd.DataFrame) -> bool:
        """Check for head and shoulders pattern"""
        try:
            highs = data['High'].rolling(window=5).max()
            peaks = argrelextrema(highs.values, np.greater_equal, order=5)[0]
            if len(peaks) >= 3:
                left, head, right = peaks[-3:]
                if (highs.iloc[head] > highs.iloc[left] and 
                    highs.iloc[head] > highs.iloc[right] and 
                    abs(highs.iloc[left] - highs.iloc[right]) / highs.iloc[left] < 0.02):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking head and shoulders: {str(e)}")
            return False

    def _check_inverse_head_shoulders(self, data: pd.DataFrame) -> bool:
        """Check for inverse head and shoulders pattern"""
        try:
            lows = data['Low'].rolling(window=5).min()
            troughs = argrelextrema(lows.values, np.less_equal, order=5)[0]
            if len(troughs) >= 3:
                left, head, right = troughs[-3:]
                if (lows.iloc[head] < lows.iloc[left] and 
                    lows.iloc[head] < lows.iloc[right] and 
                    abs(lows.iloc[left] - lows.iloc[right]) / lows.iloc[left] < 0.02):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking inverse head and shoulders: {str(e)}")
            return False

    def _calculate_ichimoku_score(self, ichimoku: Dict) -> float:
        """Calculate Ichimoku score"""
        try:
            signals = ichimoku['signals']
            primary_signal = signals['primary_signal']
            
            # Base score from primary signal
            base_score = {
                'strong_bullish': 100,
                'bullish': 75,
                'neutral': 50,
                'bearish': 25,
                'strong_bearish': 0
            }.get(primary_signal, 50)
            
            # Adjust for trend strength
            trend_strength = min(1.2, 1 + (signals['trend_strength'] / 100))
            return base_score * trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku score: {str(e)}")
            return 50

    def _calculate_lstm_score(self, data: pd.DataFrame) -> float:
        """Calculate LSTM score"""
        try:
            # Set random seeds for reproducibility
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # Prepare data - convert to numpy array first
            close_values = data['Close'].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_values.reshape(-1, 1))
            
            # Create sequences for training
            sequence_length = 60
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X = np.array(X)
            y = np.array(y)
            
            # Create model with fixed initialization using Input layer
            inputs = tf.keras.Input(shape=(sequence_length, 1))
            x = LSTM(units=50, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(units=50, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = LSTM(units=50)(x)
            x = Dropout(0.2)(x)
            outputs = Dense(units=1)(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Use more epochs and early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=50, batch_size=32, 
                     callbacks=[early_stopping],
                     verbose=0)
            
            # Prepare data for prediction
            last_sequence = scaled_data[-sequence_length:]
            current_sequence = last_sequence.reshape((1, sequence_length, 1))
            
            # Make predictions with proper type conversion
            future_predictions = []
            for _ in range(30):
                predictions = model.predict(current_sequence, verbose=0)
                # Extract single prediction value safely
                pred_value = predictions.ravel()[0] if predictions.size > 0 else 0.0
                next_pred = float(pred_value)
                future_predictions.append(next_pred)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred
            
            # Convert predictions to numpy array
            future_predictions = np.array(future_predictions, dtype=np.float64).reshape(-1, 1)
            
            # Inverse transform predictions
            future_predictions = scaler.inverse_transform(future_predictions)
            
            # Calculate confidence intervals
            last_price = float(data['Close'].iloc[-1])
            price_std = float(data['Close'].iloc[-sequence_length:].std())
            
            # Create confidence intervals
            confidence_intervals = {
                '95%': [
                    [float(x) for x in (future_predictions - 1.96 * price_std).flatten()],
                    [float(x) for x in (future_predictions + 1.96 * price_std).flatten()]
                ],
                '68%': [
                    [float(x) for x in (future_predictions - price_std).flatten()],
                    [float(x) for x in (future_predictions + price_std).flatten()]
                ]
            }
            
            # Get market condition and signals first
            market_condition = self.analyze_market_condition(data)
            signals = self.calculate_indicators(data)  # Get signals here
            tech_score = self.calculate_technical_score(data, signals)  # Calculate score

            # Calculate trend metrics
            final_pred = future_predictions[-1, 0]
            trend_direction = 'bullish' if final_pred > last_price else 'bearish'
            price_change = (final_pred - last_price) / last_price * 100

            # Adjust prediction based on technical signals
            final_prediction = self.adjust_prediction_confidence(
                price_change,
                signals['indicators'],  # Pass the indicators part of signals
                tech_score
            )

            return final_prediction

        except Exception as e:
            logger.error(f"Error calculating lstm_score: {str(e)}")
            return 0

    def _interpret_rsi(self, value: float) -> str:
        if value > 70: return 'overbought'
        if value < 30: return 'oversold'
        if value > 50: return 'bullish'
        return 'bearish'

    def _interpret_adx(self, value: float) -> str:
        if value > 25: return 'strong'
        if value > 20: return 'moderate'
        return 'weak'

    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """Calculate trend strength metrics"""
        try:
            # Get numpy arrays for calculations
            closes = data['Close'].values.flatten()[-20:]  # Get last 20 values
            
            # Calculate basic metrics
            price_range = np.ptp(closes)
            avg_price = np.mean(closes)
            volatility = np.std(closes) / avg_price
            
            # Calculate trend consistency
            returns = np.diff(closes) / closes[:-1]  # This ensures matching lengths
            direction_changes = np.sum(np.diff(returns > 0) != 0)
            consistency = 1 - (direction_changes / (len(returns) - 1))
            
            # Calculate overall strength (0-100)
            strength = (
                (consistency * 40) +  # Weight consistency more
                (min(1, price_range / avg_price) * 30) +  # Normalize range
                (max(0, 1 - volatility) * 30)  # Lower volatility is better
            )
            
            return {
                'strength': float(strength),
                'consistency': float(consistency),
                'volatility': float(volatility)
            }
            
        except Exception as e:
            print(f"Error calculating trend strength: {str(e)}")
            return {
                'strength': 0,
                'consistency': 0,
                'volatility': 0
            }

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction from a series"""
        try:
            # Get last 20 values and convert to numpy array
            values = series.tail(20).values
            if len(values) < 20:
                return 'unknown'
            
            # Calculate slope using linear regression
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Determine trend
            if slope > 0:
                return 'up'
            elif slope < 0:
                return 'down'
            return 'sideways'
            
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 'unknown'

    def _analyze_volume_trend(self, data: pd.DataFrame) -> int:
        """Analyze volume trend - returns positive for increasing, negative for decreasing"""
        recent_volume = data['Volume'].tail(20)
        return 1 if recent_volume.iloc[-1] > recent_volume.mean() else -1

    def _check_ma_alignment(self, data: pd.DataFrame) -> bool:
        """Check if moving averages are properly aligned for trend"""
        try:
            ma20 = ta.trend.sma_indicator(data['Close'], 20).iloc[-1]
            ma50 = ta.trend.sma_indicator(data['Close'], 50).iloc[-1]
            ma200 = ta.trend.sma_indicator(data['Close'], 200).iloc[-1]
            
            # Convert to float values before comparison
            ma20 = float(ma20)
            ma50 = float(ma50)
            ma200 = float(ma200)
            
            return ma20 > ma50 > ma200
            
        except Exception as e:
            print(f"Error in MA alignment check: {str(e)}")
            return False

    def _count_higher_highs(self, high: pd.Series, lookback: int = 20) -> int:
        """
        Counts the number of higher highs in recent price action
        Returns a count between -3 and 3
        """
        highs = high.tail(lookback)
        count = 0
        
        # Compare each high with previous 5 highs
        for i in range(5, len(highs)):
            current_high = highs.iloc[i]
            prev_highs = highs.iloc[i-5:i]
            
            if current_high > prev_highs.max():
                count += 1
            elif current_high < prev_highs.min():
                count -= 1
            
        return max(-3, min(3, count))  # Limit to range [-3, 3]

    def _count_higher_lows(self, low: pd.Series, lookback: int = 20) -> int:
        """
        Counts the number of higher lows in recent price action
        Returns a count between -3 and 3
        """
        lows = low.tail(lookback)
        count = 0
        
        # Compare each low with previous 5 lows
        for i in range(5, len(lows)):
            current_low = lows.iloc[i]
            prev_lows = lows.iloc[i-5:i]
            
            if current_low > prev_lows.max():
                count += 1
            elif current_low < prev_lows.min():
                count -= 1
            
        return max(-3, min(3, count))  # Limit to range [-3, 3]

    def _check_reversal_pattern(self, data: pd.DataFrame) -> str:
        """
        Checks for potential reversal patterns
        Returns: 'bullish', 'bearish', or 'none'
        """
        try:
            # Get recent price action
            recent = data.tail(5)
            
            # Calculate price movement
            price_change = (recent['Close'].iloc[-1] - recent['Open'].iloc[-1]) / recent['Open'].iloc[-1]
            prev_trend = (recent['Close'].iloc[-2] - recent['Close'].iloc[-5]) / recent['Close'].iloc[-5]
            
            # Check volume
            volume_increase = recent['Volume'].iloc[-1] > recent['Volume'].iloc[-5:].mean()
            
            # Bullish reversal conditions
            if prev_trend < -0.02 and price_change > 0.01 and volume_increase:
                return 'bullish'
            
            # Bearish reversal conditions
            if prev_trend > 0.02 and price_change < -0.01 and volume_increase:
                return 'bearish'
            
            return 'none'
            
        except Exception as e:
            print(f"Error checking reversal pattern: {str(e)}")
            return 'none'

    def _interpret_mfi(self, value: float) -> str:
        """Interpret Money Flow Index values"""
        if value > 80:
            return 'overbought'
        if value < 20:
            return 'oversold'
        if value > 50:
            return 'bullish'
        return 'bearish'

    def _calculate_price_consistency(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculates price movement consistency
        """
        try:
            # Get recent closing prices as numpy array
            closes = data['Close'].tail(window).values
            
            # Calculate daily returns using numpy
            returns = np.diff(closes) / closes[:-1]
            
            # Calculate direction changes using numpy
            direction_changes = np.sum(np.diff(returns > 0) != 0)
            
            # Calculate consistency score
            consistency = 1 - (direction_changes / (len(returns) - 1))
            
            # Calculate trend strength
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            y_pred = slope * x + intercept
            r_squared = 1 - np.sum((closes - y_pred)**2) / np.sum((closes - np.mean(closes))**2)
            
            # Final score
            final_score = float((consistency + r_squared) / 2)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"Error in price consistency: {str(e)}")
            return 0.0

    def adjust_prediction_confidence(self, prediction: float, signals: Dict, tech_score: float) -> float:
        """
        Adjust LSTM predictions based on technical signals and score
        """
        # Reduce prediction magnitude when signals conflict
        if tech_score < 40:
            prediction *= 0.5
        elif tech_score < 60:
            prediction *= 0.75
        
        # Cap extreme predictions
        max_prediction = 25.0
        if abs(prediction) > max_prediction:
            prediction = max_prediction * (1 if prediction > 0 else -1)
        
        # Adjust based on signal alignment
        all_signals = [
            signals['moving_averages']['ma_trend'],
            signals['momentum']['rsi_signal'],
            signals['trend']['macd_trend']
        ]
        
        if all(s == all_signals[0] for s in all_signals):
            # Strong signal alignment - increase confidence
            prediction *= 1.2
        elif all_signals.count('bullish') <= 1 and all_signals.count('bearish') <= 1:
            # Mixed signals - reduce confidence
            prediction *= 0.6
        
        return prediction

    def analyze_market_condition(self, data: pd.DataFrame) -> Dict:
        """
        Analyze overall market condition to adjust predictions
        """
        volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        trend_strength = self._calculate_trend_strength(data)
        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].tail(5).mean()
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'volume_trend': recent_volume / avg_volume,
            'market_condition': 'stable' if volatility < 0.2 else 'volatile'
        }

    def predict_price_movement(self, data: pd.DataFrame) -> Dict:
        """Ensemble prediction using multiple technical indicators"""
        try:
            # Calculate all indicators
            signals = self.calculate_indicators(data)
            trend = self.analyze_trend(data)
            support_resistance = self.find_support_resistance(data)
            patterns = self._check_patterns(data)
            
            # Get current price and calculate basic metrics
            current_price = data['Close'].iloc[-1]
            volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
            
            # Combine predictions from different methods
            predictions = {
                'trend_following': self._trend_following_prediction(signals, trend),
                'momentum': self._momentum_prediction(signals),
                'pattern': self._pattern_based_prediction(patterns),
                'support_resistance': self._sr_based_prediction(support_resistance, current_price)
            }
            
            # Weight and combine predictions
            weights = {
                'trend_following': 0.35,
                'momentum': 0.25,
                'pattern': 0.20,
                'support_resistance': 0.20
            }
            
            final_prediction = 0
            for method, pred in predictions.items():
                final_prediction += pred * weights[method]
            
            # Calculate confidence based on indicator agreement
            predictions_list = list(predictions.values())
            confidence = 1 - np.std(predictions_list) / (max(abs(min(predictions_list)), abs(max(predictions_list))) + 1e-6)
            
            return {
                'predictions': predictions,
                'trend': {
                    'direction': 'bullish' if final_prediction > 0 else 'bearish',
                    'predicted_change_percent': float(final_prediction),
                    'confidence': float(confidence)
                },
                'market_condition': {
                    'volatility': float(volatility),
                    'trend_strength': trend['strength']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            return {
                'error': str(e),
                'predictions': None,
                'trend': {
                    'direction': 'unknown', 
                    'predicted_change_percent': 0,
                    'confidence': 0.0  # Add confidence even in error case
                },
                'market_condition': {
                    'volatility': 0, 
                    'trend_strength': 0
                }
            }

    def _trend_following_prediction(self, signals: Dict, trend: Dict) -> float:
        """Generate prediction based on trend following indicators"""
        ma = signals['indicators']['moving_averages']
        
        # Calculate prediction based on MA crossovers and trend strength
        if ma['ma_trend'] == 'bullish':
            base_prediction = 2.0
        elif ma['ma_trend'] == 'bearish':
            base_prediction = -2.0
        else:
            base_prediction = 0.0
            
        # Adjust based on trend strength
        trend_multiplier = trend['strength'] / 50  # Normalize to [-2, 2] range
        return base_prediction * trend_multiplier

    def _momentum_prediction(self, signals: Dict) -> float:
        """Generate prediction based on momentum indicators"""
        try:
            # Get momentum and trend indicators
            momentum = signals['indicators']['momentum']
            trend = signals['indicators']['trend']
            
            # RSI-based prediction
            rsi = momentum['rsi']
            if rsi > 70:
                rsi_pred = -1.5  # Overbought
            elif rsi < 30:
                rsi_pred = 1.5   # Oversold
            else:
                rsi_pred = (rsi - 50) / 10  # Scaled prediction
                
            # MACD-based prediction
            macd_pred = 1.0 if trend['macd_trend'] == 'bullish' else -1.0
            
            return (rsi_pred + macd_pred) / 2
            
        except Exception as e:
            logger.error(f"Error in momentum prediction: {str(e)}")
            return 0.0

    def _pattern_based_prediction(self, patterns: Dict) -> float:
        """Generate prediction based on chart patterns"""
        bullish_patterns = ['double_bottom', 'inverse_head_shoulders', 'bullish_flag']
        bearish_patterns = ['double_top', 'head_shoulders', 'bearish_flag']
        
        bullish_count = sum(1 for p in bullish_patterns if patterns.get(p, False))
        bearish_count = sum(1 for p in bearish_patterns if patterns.get(p, False))
        
        if bullish_count > bearish_count:
            return 2.0
        elif bearish_count > bullish_count:
            return -2.0
        return 0.0

    def _sr_based_prediction(self, sr_levels: Dict, current_price: float) -> float:
        """Generate prediction based on support and resistance levels"""
        support = min(sr_levels['support']['levels'], default=current_price)
        resistance = max(sr_levels['resistance']['levels'], default=current_price)
        
        support_distance = (current_price - support) / current_price
        resistance_distance = (resistance - current_price) / current_price
        
        if support_distance < 0.02:  # Close to support
            return 1.5
        elif resistance_distance < 0.02:  # Close to resistance
            return -1.5
        
        # Return scaled prediction based on relative distances
        return (resistance_distance - support_distance) * 100

    def analyze_stock_data(self, hist: pd.DataFrame) -> Dict:
        """Analyzes stock using provided historical data"""
        if len(hist) < self.indicators[self.timeframe]['ma_long']:
            return {'error': 'Insufficient historical data'}
        
        signals = self.calculate_indicators(hist)
        levels = self.find_support_resistance(hist)
        fib_levels = self.calculate_fibonacci_levels(hist)
        trend = self.analyze_trend(hist)
        predictions = self.predict_price_movement(hist)
        score = self.calculate_technical_score(hist, signals)
        
        return {
            'signals': signals,
            'support_resistance': levels,
            'fibonacci_levels': fib_levels,
            'trend': trend,
            'price_predictions': predictions,
            'technical_score': score
        }

    def analyze_price_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze candlestick and chart patterns"""
        try:
            # Get recent data
            recent = data.tail(20)
            
            # Candlestick patterns
            doji = abs(recent['Open'] - recent['Close']) <= (recent['High'] - recent['Low']) * 0.1
            hammer = (recent['Low'] - recent['Close']) >= 2 * (recent['Open'] - recent['Close'])
            shooting_star = (recent['High'] - recent['Close']) >= 2 * (recent['Close'] - recent['Open'])
            
            # Chart patterns
            double_top = self._check_double_top(data)
            double_bottom = self._check_double_bottom(data)
            head_shoulders = self._check_head_shoulders(data)
            
            return {
                'candlestick_patterns': {
                    'doji': bool(doji.iloc[-1]),
                    'hammer': bool(hammer.iloc[-1]),
                    'shooting_star': bool(shooting_star.iloc[-1])
                },
                'chart_patterns': {
                    'double_top': double_top,
                    'double_bottom': double_bottom,
                    'head_shoulders': head_shoulders
                }
            }
        except Exception as e:
            print(f"Error analyzing patterns: {str(e)}")
            return {}

    def analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyze price volatility and momentum"""
        try:
            # Calculate volatility metrics
            returns = data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # ATR for volatility
            atr = ta.volatility.AverageTrueRange(
                data['High'], data['Low'], data['Close']
            ).average_true_range()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            
            return {
                'metrics': {
                    'volatility': float(volatility),
                    'atr': float(atr.iloc[-1]),
                    'bb_width': float(bb.bollinger_wband().iloc[-1])
                },
                'signals': {
                    'high_volatility': volatility > 0.3,
                    'expanding_bb': bb.bollinger_wband().diff().iloc[-1] > 0,
                    'price_above_bb': data['Close'].iloc[-1] > bb.bollinger_hband().iloc[-1]
                }
            }
        except Exception as e:
            print(f"Error analyzing volatility: {str(e)}")
            return {}

    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict:
        """Enhanced volume analysis"""
        try:
            # Volume metrics
            avg_volume = data['Volume'].mean()
            recent_volume = data['Volume'].tail(5).mean()
            
            # Volume-price correlation
            price_changes = data['Close'].pct_change()
            volume_changes = data['Volume'].pct_change()
            correlation = price_changes.corr(volume_changes)
            
            # Volume trends
            increasing_volume = recent_volume > avg_volume * 1.2
            decreasing_volume = recent_volume < avg_volume * 0.8
            
            return {
                'metrics': {
                    'avg_volume': float(avg_volume),
                    'recent_volume': float(recent_volume),
                    'volume_price_correlation': float(correlation)
                },
                'signals': {
                    'increasing_volume': bool(increasing_volume),
                    'decreasing_volume': bool(decreasing_volume),
                    'volume_confirms_price': (
                        correlation > 0.3 if price_changes.iloc[-1] > 0 
                        else correlation < -0.3
                    )
                }
            }
        except Exception as e:
            print(f"Error analyzing volume: {str(e)}")
            return {}

    def get_comprehensive_analysis(self, data: pd.DataFrame) -> Dict:
        """Get comprehensive technical analysis"""
        try:
            # Basic indicators
            signals = self.calculate_indicators(data)
            
            # Additional analyses
            patterns = self.analyze_price_patterns(data)
            volatility = self.analyze_volatility(data)
            volume = self.analyze_volume_profile(data)
            support_resistance = self.find_support_resistance(data)
            trend = self.analyze_trend(data)
            predictions = self.predict_price_movement(data)
            
            # Combine all analyses
            return {
                'signals': signals,
                'patterns': patterns,
                'volatility': volatility,
                'volume': volume,
                'support_resistance': support_resistance,
                'trend': trend,
                'predictions': predictions,
                'summary': {
                    'overall_trend': trend['trend'],
                    'risk_level': 'high' if volatility['metrics']['volatility'] > 0.3 else 'medium',
                    'trading_signals': self._generate_trading_signals(
                        signals, patterns, volatility, volume
                    )
                }
            }
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            return {}

    def _generate_trading_signals(self, signals: Dict, patterns: Dict, 
                                volatility: Dict, volume: Dict) -> Dict:
        """Generate trading signals from all indicators"""
        try:
            # Collect bullish signals
            bullish_signals = [
                signals['indicators']['moving_averages']['ma_trend'] == 'bullish',
                signals['indicators']['momentum']['rsi_signal'] == 'oversold',
                signals['indicators']['trend']['macd_trend'] == 'bullish',
                patterns['chart_patterns']['double_bottom'],
                volume['signals']['volume_confirms_price']
            ]
            
            # Collect bearish signals
            bearish_signals = [
                signals['indicators']['moving_averages']['ma_trend'] == 'bearish',
                signals['indicators']['momentum']['rsi_signal'] == 'overbought',
                signals['indicators']['trend']['macd_trend'] == 'bearish',
                patterns['chart_patterns']['double_top'],
                not volume['signals']['volume_confirms_price']
            ]
            
            # Calculate signal strengths
            bullish_strength = sum(bullish_signals) / len(bullish_signals) * 100
            bearish_strength = sum(bearish_signals) / len(bearish_signals) * 100
            
            return {
                'primary_signal': 'bullish' if bullish_strength > bearish_strength else 'bearish',
                'signal_strength': max(bullish_strength, bearish_strength),
                'confidence': 'high' if abs(bullish_strength - bearish_strength) > 40 else 'medium',
                'suggested_action': self._get_suggested_action(
                    bullish_strength, bearish_strength, volatility['metrics']['volatility']
                )
            }
        except Exception as e:
            print(f"Error generating trading signals: {str(e)}")
            return {}

    def calculate_comprehensive_score(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical score using all available metrics"""
        try:
            # Get all analyses
            analysis = self.get_comprehensive_analysis(data)
            
            # Component weights
            weights = {
                'trend': 0.15,          # 15% - Traditional trend analysis
                'momentum': 0.15,       # 15% - Momentum indicators (RSI, MACD)
                'support_resist': 0.10, # 10% - Support/Resistance levels
                'volatility': 0.10,     # 10% - Volatility metrics
                'volume': 0.15,         # 15% - Volume analysis
                'patterns': 0.05,       # 5% - Chart patterns
                'ichimoku': 0.15,       # 15% - Ichimoku Cloud analysis
                'lstm': 0.15           # 15% - LSTM predictions
            }
            
            scores = {}
            
            # 1. Trend Score (0-100)
            trend_score = (
                (100 if analysis['trend']['trend'].startswith('strong_') else 
                 75 if analysis['trend']['trend'] in ['uptrend', 'downtrend'] else 50) +
                analysis['trend']['strength']
            ) / 2
            
            # 2. Momentum Score (0-100)
            signals = analysis['signals']['indicators']
            momentum_score = (
                (100 if signals['momentum']['rsi_signal'] in ['oversold', 'overbought'] else 60) +
                (100 if abs(signals['trend']['macd']) > abs(signals['trend']['macd_signal']) else 60) +
                (100 if signals['momentum']['stochastic'] > 80 or signals['momentum']['stochastic'] < 20 else 60)
            ) / 3
            
            # 3. Support/Resistance Score (0-100)
            sr_levels = analysis['support_resistance']
            nearest_support = sr_levels['support']['levels'][0] if sr_levels['support']['levels'] else 0
            nearest_resist = sr_levels['resistance']['levels'][0] if sr_levels['resistance']['levels'] else float('inf')
            current_price = sr_levels['current_price']
            
            sr_score = (
                (sr_levels['support']['strength'][0] if sr_levels['support']['strength'] else 0) +
                (sr_levels['resistance']['strength'][0] if sr_levels['resistance']['strength'] else 0)
            ) / 2
            
            # 4. Volatility Score (0-100)
            vol = analysis['volatility']['metrics']
            vol_score = (
                (100 if vol['volatility'] < 0.2 else 
                 70 if vol['volatility'] < 0.3 else 
                 40 if vol['volatility'] < 0.4 else 20) +
                (100 if not analysis['volatility']['signals']['high_volatility'] else 50)
            ) / 2
            
            # 5. Volume Score (0-100)
            vol_analysis = analysis['volume']
            volume_score = (
                (100 if vol_analysis['signals']['volume_confirms_price'] else 50) +
                (100 if vol_analysis['metrics']['volume_price_correlation'] > 0.5 else 
                 70 if vol_analysis['metrics']['volume_price_correlation'] > 0.3 else 50) +
                (100 if vol_analysis['signals']['increasing_volume'] else 60)
            ) / 3
            
            # 6. Pattern Score (0-100)
            patterns = analysis['patterns']
            pattern_score = (
                (100 if any(patterns['candlestick_patterns'].values()) else 60) +
                (100 if any(patterns['chart_patterns'].values()) else 60)
            ) / 2
            
            # Add Ichimoku Score (0-100)
            ichimoku_score = self.calculate_ichimoku(data)
            if 'error' not in ichimoku_score:
                ichimoku_score = ichimoku_score['signals']['trend_strength']
            else:
                ichimoku_score = 50  # Neutral score if Ichimoku fails
            
            # Add LSTM Score (0-100)
            lstm_pred = analysis['predictions']
            if 'error' not in lstm_pred:
                price_change = lstm_pred['trend']['predicted_change_percent']
                confidence_intervals = lstm_pred['confidence_intervals']
                market_condition = lstm_pred['market_condition']
                
                # Calculate LSTM score based on:
                # 1. Prediction strength (0-40 points)
                pred_strength = min(40, abs(price_change) * 2)
                
                # 2. Confidence level (0-30 points)
                conf_range = abs(confidence_intervals['68%'][1][0] - confidence_intervals['68%'][0][0])
                conf_score = 30 * (1 - min(1, conf_range / abs(price_change)))
                
                # 3. Market condition alignment (0-30 points)
                market_alignment = 30 if (
                    (price_change > 0 and market_condition['trend_strength']['strength'] > 60) or
                    (price_change < 0 and market_condition['trend_strength']['strength'] < 40)
                ) else 15
                
                lstm_score = pred_strength + conf_score + market_alignment
            else:
                lstm_score = 50  # Neutral score if LSTM fails

            # Update scores dictionary
            scores = {
                'trend_score': float(trend_score),
                'momentum_score': float(momentum_score),
                'support_resistance_score': float(sr_score),
                'volatility_score': float(vol_score),
                'volume_score': float(volume_score),
                'pattern_score': float(pattern_score),
                'ichimoku_score': float(ichimoku_score),
                'lstm_score': float(lstm_score)
            }
            
            # Calculate final weighted score
            final_score = sum(
                scores[f"{k}_score"] * v 
                for k, v in weights.items()
            )
            
            return {
                'total_score': float(final_score),
                'component_scores': scores,
                'weights': weights,
                'interpretation': self._interpret_technical_score(final_score, scores)
            }
            
        except Exception as e:
            print(f"Error calculating comprehensive score: {str(e)}")
            return {
                'total_score': 0,
                'component_scores': {},
                'weights': weights,
                'interpretation': 'Error calculating score'
            }

    def _interpret_technical_score(self, total_score: float, component_scores: Dict) -> str:
        """Interpret the technical score and provide detailed analysis"""
        try:
            strength = (
                'very bearish' if total_score < 30 else
                'bearish' if total_score < 45 else
                'neutral' if total_score < 55 else
                'bullish' if total_score < 70 else
                'very bullish'
            )
            
            # Find strongest and weakest components
            strongest = max(component_scores.items(), key=lambda x: x[1])
            weakest = min(component_scores.items(), key=lambda x: x[1])
            
            return {
                'overall_rating': strength,
                'score': float(total_score),
                'summary': f"Technical analysis indicates {strength} outlook",
                'strongest_factor': f"{strongest[0].replace('_score', '')}: {strongest[1]:.1f}",
                'weakest_factor': f"{weakest[0].replace('_score', '')}: {weakest[1]:.1f}",
                'confidence': 'high' if abs(strongest[1] - weakest[1]) < 30 else 'medium'
            }
            
        except Exception as e:
            print(f"Error interpreting score: {str(e)}")
            return "Error interpreting technical score"

    def calculate_ichimoku(self, data: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud indicators"""
        try:
            # Calculate Ichimoku components
            high_prices = data['High']
            low_prices = data['Low']
            closing_prices = data['Close']
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = high_prices.rolling(window=9).max()
            period9_low = low_prices.rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = high_prices.rolling(window=26).max()
            period26_low = low_prices.rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = high_prices.rolling(window=52).max()
            period52_low = low_prices.rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Current closing price shifted back 26 periods
            chikou_span = closing_prices.shift(-26)
            
            # Get current values
            current_price = closing_prices.iloc[-1]
            current_tenkan = tenkan_sen.iloc[-1]
            current_kijun = kijun_sen.iloc[-1]
            current_senkou_a = senkou_span_a.iloc[-1]
            current_senkou_b = senkou_span_b.iloc[-1]
            
            # Determine cloud status
            cloud_color = 'green' if current_senkou_a > current_senkou_b else 'red'
            price_relative_cloud = (
                'above' if current_price > max(current_senkou_a, current_senkou_b) else
                'below' if current_price < min(current_senkou_a, current_senkou_b) else
                'in_cloud'
            )
            
            # Generate trading signals
            trend_strength = abs(current_senkou_a - current_senkou_b) / current_price * 100
            
            signals = {
                'primary_signal': (
                    'strong_bullish' if (price_relative_cloud == 'above' and 
                                       current_tenkan > current_kijun and
                                       cloud_color == 'green') else
                    'bullish' if (price_relative_cloud == 'above' or 
                                (current_tenkan > current_kijun and cloud_color == 'green')) else
                    'strong_bearish' if (price_relative_cloud == 'below' and 
                                       current_tenkan < current_kijun and
                                       cloud_color == 'red') else
                    'bearish' if (price_relative_cloud == 'below' or 
                                (current_tenkan < current_kijun and cloud_color == 'red')) else
                    'neutral'
                ),
                'trend_strength': float(trend_strength),
                'cloud_color': cloud_color,
                'price_position': price_relative_cloud
            }
            
            return {
                'current_values': {
                    'tenkan_sen': float(current_tenkan),
                    'kijun_sen': float(current_kijun),
                    'senkou_span_a': float(current_senkou_a),
                    'senkou_span_b': float(current_senkou_b)
                },
                'signals': signals,
                'cloud_thickness': float(abs(current_senkou_a - current_senkou_b)),
                'trend_confirmation': current_tenkan > current_kijun and cloud_color == 'green'
            }
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {str(e)}")
            return None

    def _safe_float_conversion(self, value) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except Exception as e:
            logger.error(f"Error converting to float: {str(e)}")
            return 0.0

# Add this at the bottom of the file
if __name__ == "__main__":
    def test_run():
        try:
            timeframes = ['medium', 'long']
            test_tickers = ['AAPL', 'MSFT', 'GOOGL']
            
            for ticker in test_tickers:
                print(f"\n{'='*60}")
                print(f"{ticker} Analysis")
                print(f"{'='*60}")
                
                for timeframe in timeframes:
                    analyzer = TechnicalAnalyzer(timeframe=timeframe)
                    logger.info(f"\nAnalyzing {ticker} ({timeframe} term)...")
                    
                    analysis = analyzer.analyze_stock(ticker)
                    
                    if 'error' in analysis:
                        print(f"\n{timeframe.upper()} TERM - Error: {analysis['error']}")
                        continue
                    
                    print(f"\n{timeframe.upper()} TERM ANALYSIS:")
                    print("-" * 50)
                    
                    # Get current price
                    current_price = analysis['signals']['current_price']
                    print(f"Current Price: ${current_price:.2f}")
                    
                    # Print Entry/Exit Levels
                    sr = analysis['support_resistance']
                    supports = sr['support']['levels']
                    resistances = sr['resistance']['levels']
                    
                    print("\nTrading Levels:")
                    print("-" * 20)
                    
                    # Strong Sell/Resistance Levels
                    if resistances:
                        print(f"Strong Sell    : ${resistances[0]:.2f}")
                        if len(resistances) > 1:
                            print(f"Weak Sell      : ${resistances[1]:.2f}")
                    
                    # Buy/Support Levels
                    if supports:
                        print(f"Strong Buy     : ${supports[0]:.2f}")
                        if len(supports) > 1:
                            print(f"Weak Buy       : ${supports[1]:.2f}")
                    
                    # Print trend information
                    trend = analysis['predictions']['final']
                    print(f"\nTrend Direction: {trend['direction'].upper()}")
                    print(f"Expected Move  : {trend['predicted_change_percent']:.2f}%")
                    print(f"Confidence     : {trend['confidence']:.2f}")
                    
                    # Print market condition
                    market = analysis['market_condition']
                    print(f"\nMarket Conditions:")
                    print(f"Volatility    : {market['volatility']:.2f}")
                    print(f"Trend Strength: {market['trend_strength']:.2f}")
                    
                    # Print technical score and rating
                    tech_score = analysis['technical_score']
                    print(f"\nOverall Analysis:")
                    print(f"Technical Score: {tech_score['total_score']:.2f}")
                    print(f"Rating        : {tech_score['interpretation']['overall_rating'].upper()}")
                    print(f"Confidence    : {tech_score['interpretation']['confidence'].upper()}")
                    
                    # Trading Recommendation
                    score = tech_score['total_score']
                    if score >= 70:
                        action = "STRONG BUY"
                    elif score >= 60:
                        action = "BUY"
                    elif score <= 30:
                        action = "STRONG SELL"
                    elif score <= 40:
                        action = "SELL"
                    else:
                        action = "HOLD"
                    
                    print(f"\nRecommendation: {action}")
                    print("-" * 50)
            
        except Exception as e:
            logger.error(f"Test run failed: {str(e)}")
            raise

    # Run the test
    test_run()