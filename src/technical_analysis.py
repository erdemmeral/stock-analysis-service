import logging
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from typing import Dict
from scipy.signal import argrelextrema
from src.config import TIMEFRAME_CONFIGS

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self, timeframe: str = 'medium'):
        self.timeframe = timeframe
        self.config = TIMEFRAME_CONFIGS[timeframe]
        self.periods = {
            'ma_short': self.config['ma_short'],
            'ma_long': self.config['ma_long'],
            'rsi': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volatility': 20  # Days for volatility calculation
        }

    def analyze_stock(self, ticker: str) -> Dict:
        """Main analysis function with timeframe-specific requirements"""
        try:
            analyses = {}
            timeframe_scores = {}
            
            # Different minimum periods per timeframe
            min_periods = {
                'short': 30,   # 30 days for short-term
                'medium': 90,  # 90 days for medium-term
                'long': 200    # 200 days for long-term
            }
            
            for tf in ['short', 'medium', 'long']:
                config = TIMEFRAME_CONFIGS[tf]
                stock = yf.Ticker(ticker)
                hist = stock.history(period=config['period'], interval=config['interval'])
                
                # Check minimum required periods
                if len(hist) < min_periods[tf]:
                    logger.warning(f"Insufficient data for {tf} timeframe: {len(hist)} < {min_periods[tf]}")
                    continue
                
                analysis = self._analyze_timeframe(hist, config)
                analyses[tf] = analysis
                timeframe_scores[tf] = analysis['technical_score']['total']
            
            if not analyses:
                return self.get_empty_analysis()
            
            # Get the primary timeframe analysis
            primary_analysis = analyses.get(self.timeframe, analyses['medium'])
            
            # Add all timeframe scores
            primary_analysis['timeframe_scores'] = timeframe_scores
            
            # Calculate weighted average score using weights from config
            total_weight = sum(TIMEFRAME_CONFIGS[tf]['weight'] for tf in timeframe_scores.keys())
            weighted_score = sum(
                timeframe_scores[tf] * TIMEFRAME_CONFIGS[tf]['weight']
                for tf in timeframe_scores.keys()
            ) / total_weight if total_weight > 0 else 50
            
            # Update technical score with combined score
            primary_analysis['technical_score'].update({
                'total': weighted_score,
                'timeframes': timeframe_scores
            })
            
            return primary_analysis

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return self.get_empty_analysis()

    def _analyze_timeframe(self, hist: pd.DataFrame, config: Dict) -> Dict:
        """Analyze a single timeframe with improved signal handling"""
        try:
            # Calculate indicators with proper NaN handling
            signals = {
                'current_price': float(hist['Close'].iloc[-1]),
                'momentum': self._calculate_momentum(hist, config['period']),  # Pass period
                'trend': self._calculate_trend(hist, config['ma_short'], config['ma_long']),  # Pass MA periods
                'volume': self._calculate_volume(hist),
                'moving_averages': self._calculate_ma_signals(hist, config),
                'volatility': self._calculate_volatility(hist)
            }
            
            # Calculate support/resistance with dynamic windows
            volatility = signals['volatility']['daily']
            window_size = max(20, int(20 * (1 + volatility)))
            support_resistance = self._calculate_support_resistance(hist, window_size)
            
            # Calculate technical score with timeframe-specific adjustments
            technical_score = self._calculate_score(signals, config)
            
            return {
                'signals': signals,
                'support_resistance': support_resistance,
                'technical_score': technical_score
            }
            
        except Exception as e:
            logger.error(f"Error in timeframe analysis: {e}")
            return self.get_empty_analysis()

    def _calculate_momentum(self, hist: pd.DataFrame, period: str) -> Dict:
        """Calculate momentum indicators with period-specific settings"""
        # Adjust RSI period based on timeframe
        rsi_period = 14 if period in ['1mo', '2mo'] else 21 if period == '6mo' else 30
        rsi = ta.momentum.RSIIndicator(hist['Close'], window=rsi_period).rsi()
        
        # Stochastic with adjusted periods
        stoch_period = 14 if period in ['1mo', '2mo'] else 21 if period == '6mo' else 30
        stoch = ta.momentum.StochasticOscillator(
            hist['High'], hist['Low'], hist['Close'],
            window=stoch_period
        )
        
        return {
            'rsi': float(rsi.iloc[-1]),
            'stochastic': {
                'k': float(stoch.stoch().iloc[-1]),
                'd': float(stoch.stoch_signal().iloc[-1])
            }
        }

    def _calculate_trend(self, hist: pd.DataFrame, ma_short: int, ma_long: int) -> Dict:
        """Calculate trend indicators with timeframe-specific MA periods"""
        # Calculate MACD with adjusted periods
        macd = ta.trend.MACD(
            hist['Close'],
            window_slow=ma_long,
            window_fast=ma_short,
            window_sign=9
        )
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        
        # Trim initial NaN periods
        valid_index = macd_line.first_valid_index()
        if valid_index:
            macd_line = macd_line[valid_index:]
            signal_line = signal_line[valid_index:]
        
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        
        # Determine trend direction and strength
        trend_strength = abs(current_macd - current_signal)
        if current_macd > current_signal:
            direction = 'strong_bullish' if trend_strength > 0.5 else 'bullish'
        else:
            direction = 'strong_bearish' if trend_strength > 0.5 else 'bearish'
        
        return {
            'direction': direction,
            'strength': trend_strength,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_trend': 'bullish' if current_macd > current_signal else 'bearish'
        }

    def _calculate_score(self, signals: Dict, config: Dict) -> Dict:
        """Calculate weighted technical score with timeframe-specific adjustments"""
        # RSI scoring
        rsi = signals['momentum']['rsi']
        rsi_score = (
            90 if rsi < 35 else
            80 if 35 <= rsi < 40 else
            70 if 40 <= rsi < 45 else
            60 if 45 <= rsi < 50 else
            50 if 50 <= rsi < 55 else
            40 if 55 <= rsi < 60 else
            30 if 60 <= rsi < 65 else
            20
        )
        
        # Trend scoring
        trend_score = {
            'strong_bullish': 100,
            'bullish': 75,
            'neutral': 50,
            'bearish': 25,
            'strong_bearish': 0
        }[signals['trend']['direction']]
        
        # Volume scoring
        volume_score = {
            'high': 100,
            'increasing': 75,
            'normal': 50,
            'decreasing': 25,
            'low': 0
        }[signals['volume']['profile']]
        
        # Get timeframe-specific weights
        period = config['period']
        if period in ['1mo', '2mo']:  # Short-term
            weights = {
                'momentum': 0.4,
                'trend': 0.4,
                'volume': 0.2,
                'volatility_bonus': 5 if signals['volatility']['trend'] == 'decreasing' else -5
            }
        elif period == '6mo':  # Medium-term
            weights = {
                'momentum': 0.3,
                'trend': 0.5,
                'volume': 0.2,
                'volatility_bonus': 3 if signals['volatility']['trend'] == 'stable' else -3
            }
        else:  # Long-term
            weights = {
                'momentum': 0.2,
                'trend': 0.6,
                'volume': 0.2,
                'volatility_bonus': 2 if signals['volatility']['trend'] == 'stable' else -2
            }
        
        # Calculate adjusted score
        adjusted_score = (
            rsi_score * weights['momentum'] +
            trend_score * weights['trend'] +
            volume_score * weights['volume'] +
            weights['volatility_bonus']
        )
        
        # Ensure score is within 0-100 range
        final_score = max(0, min(100, adjusted_score))
        
        return {
            'total': final_score,
            'momentum': rsi_score,
            'trend': trend_score,
            'volume': volume_score
        }

    def _calculate_volatility(self, hist: pd.DataFrame) -> Dict:
        """Calculate volatility metrics with rolling Z-score analysis"""
        returns = hist['Close'].pct_change()
        
        # Daily volatility
        daily_vol = returns.std()
        
        # Annualized volatility
        annual_vol = daily_vol * np.sqrt(252)
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.periods['volatility']).std()
        
        # Calculate rolling Z-score for volatility trend detection
        rolling_mean = rolling_vol.rolling(window=20).mean()
        rolling_std = rolling_vol.rolling(window=20).std()
        z_score = (rolling_vol - rolling_mean) / rolling_std
        
        # Get recent volatility trend using Z-scores
        recent_z_score = z_score.iloc[-1]
        
        # Determine trend based on statistical significance
        if recent_z_score > 1.645:  # 90% confidence level
            vol_trend = 'increasing'
        elif recent_z_score < -1.645:
            vol_trend = 'decreasing'
        else:
            vol_trend = 'stable'
        
        return {
            'daily': float(daily_vol),
            'annual': float(annual_vol),
            'trend': vol_trend,
            'z_score': float(recent_z_score),
            'risk_level': 'high' if annual_vol > 0.4 else \
                         'medium' if annual_vol > 0.2 else \
                         'low'
        }

    def _calculate_volume(self, hist: pd.DataFrame) -> Dict:
        """Analyze volume trend"""
        recent_vol = hist['Volume'].tail(5).mean()
        avg_vol = hist['Volume'].mean()
        
        if recent_vol > avg_vol * 1.2:
            profile = 'high'
        elif recent_vol > avg_vol:
            profile = 'increasing'
        elif recent_vol < avg_vol * 0.8:
            profile = 'decreasing'
        else:
            profile = 'normal'
            
        return {
            'profile': profile,
            'value': float(recent_vol)
        }

    def _calculate_ma_signals(self, hist: pd.DataFrame, config: Dict) -> Dict:
        """Calculate moving averages"""
        ma_short = ta.trend.SMAIndicator(hist['Close'], config['ma_short']).sma_indicator()
        ma_long = ta.trend.SMAIndicator(hist['Close'], config['ma_long']).sma_indicator()
        
        current_short = float(ma_short.iloc[-1])
        current_long = float(ma_long.iloc[-1])
        
        return {
            'trend': 'bullish' if current_short > current_long else 'bearish',
            'aligned': current_short > current_long
        }

    def _calculate_support_resistance(self, hist: pd.DataFrame, window_size: int) -> Dict:
        """Find support and resistance levels"""
        price_data = hist['Close'].values
        support = sorted(price_data[argrelextrema(price_data, np.less, order=window_size)[0]], reverse=True)
        resistance = sorted(price_data[argrelextrema(price_data, np.greater, order=window_size)[0]])
        
        return {
            'support': support[:3],
            'resistance': resistance[:3]
        }

    def get_empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'signals': {
                'current_price': 0,
                'momentum': {'rsi': 50, 'stochastic': {'k': 50, 'd': 50}},
                'trend': {
                    'direction': 'neutral',
                    'strength': 0,
                    'macd': 0,
                    'macd_signal': 0,
                    'macd_trend': 'neutral'
                },
                'volume': {'profile': 'normal', 'value': 0},
                'moving_averages': {'trend': 'neutral', 'aligned': False},
                'volatility': {
                    'daily': 0,
                    'annual': 0,
                    'trend': 'stable',
                    'risk_level': 'low'
                }
            },
            'support_resistance': {'support': [0], 'resistance': [0]},
            'technical_score': {
                'total': 50,
                'momentum': 50,
                'trend': 50,
                'volume': 50,
                'timeframes': {
                    'short': 50,
                    'medium': 50,
                    'long': 50
                }
            },
            'timeframe_scores': {
                'short': 50,
                'medium': 50,
                'long': 50
            }
        }

def run_test_analysis():
    """Test function to verify technical analysis"""
    logger.info("Starting technical analysis test...")
    
    # Initialize analyzer
    analyzer = TechnicalAnalyzer()
    
    # Test with a well-known stock
    test_ticker = "AAPL"
    
    try:
        # Get analysis
        analysis = analyzer.analyze_stock(test_ticker)
        
        print("\n=== Technical Analysis Test Results ===")
        print(f"Stock: {test_ticker}")
        
        print("\n1. Technical Scores:")
        print(f"Combined Score: {analysis['technical_score']['total']:.2f}")
        print("\nTimeframe Scores:")
        for tf, score in analysis['technical_score']['timeframes'].items():
            print(f"{tf.capitalize()}: {score:.2f}")
        
        print("\n2. Component Scores:")
        print(f"Momentum: {analysis['technical_score']['momentum']:.2f}")
        print(f"Trend: {analysis['technical_score']['trend']:.2f}")
        print(f"Volume: {analysis['technical_score']['volume']:.2f}")
        
        print("\n3. Signal Details:")
        signals = analysis['signals']
        print(f"RSI: {signals['momentum']['rsi']:.2f}")
        print(f"Trend Direction: {signals['trend']['direction']}")
        print(f"Volume Profile: {signals['volume']['profile']}")
        print(f"Risk Level: {signals['volatility']['risk_level']}")
        
        print("\n4. Support/Resistance:")
        sr = analysis['support_resistance']
        print(f"Support Levels: {', '.join([f'${x:.2f}' for x in sr['support']])}")
        print(f"Resistance Levels: {', '.join([f'${x:.2f}' for x in sr['resistance']])}")
        
        print("\n=== Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_component_tests():
    """Add component tests to verify everything is working"""
    logger.info("Starting component tests...")
    
    # Test 1: Data Fetching
    def test_data_fetching():
        try:
            stock = yf.Ticker("AAPL")
            hist = stock.history(period="3mo")
            return len(hist) > 0
        except Exception as e:
            logger.error(f"Data fetching test failed: {e}")
            return False
    
    # Test 2: Technical Indicators
    def test_technical_indicators():
        try:
            analyzer = TechnicalAnalyzer()
            analysis = analyzer.analyze_stock("AAPL")
            return all(key in analysis['signals'] for key in [
                'momentum', 'trend', 'volume', 'moving_averages', 'volatility'
            ])
        except Exception as e:
            logger.error(f"Technical indicators test failed: {e}")
            return False
    
    # Test 3: News Analysis
    def test_news_analysis():
        try:
            from src.news_analysis import NewsAnalyzer
            news_analyzer = NewsAnalyzer()
            news_analysis = news_analyzer.analyze_stock_news("AAPL")
            return all(key in news_analysis for key in [
                'news_score', 'sentiment', 'articles'
            ])
        except Exception as e:
            logger.error(f"News analysis test failed: {e}")
            return False

    # Test 4: Fundamental Analysis
    def test_fundamental_analysis():
        try:
            from src.fundamental_analysis import FundamentalAnalyzer
            fund_analyzer = FundamentalAnalyzer()
            fund_analysis = fund_analyzer.analyze_stocks()
            return isinstance(fund_analysis, tuple) and len(fund_analysis) == 2
        except Exception as e:
            logger.error(f"Fundamental analysis test failed: {e}")
            return False

    # Test 5: Integration Test
    def test_integration():
        try:
            from src.service.analysis_service import AnalysisService
            service = AnalysisService()
            import asyncio
            result = asyncio.run(service.test_integration())
            return result
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False

    # Run all tests
    tests = {
        "Data Fetching": test_data_fetching(),
        "Technical Indicators": test_technical_indicators(),
        "News Analysis": test_news_analysis(),
        "Fundamental Analysis": test_fundamental_analysis(),
        "Service Integration": test_integration()
    }
    
    # Print results
    print("\n=== Component Test Results ===")
    all_passed = True
    for test_name, passed in tests.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    return all_passed

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run both analysis and component tests
    run_test_analysis()
    run_component_tests()