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
        """Main analysis function"""
        try:
            # Get data for all timeframes
            analyses = {}
            for tf in ['short', 'medium', 'long']:
                config = TIMEFRAME_CONFIGS[tf]
                stock = yf.Ticker(ticker)
                hist = stock.history(period=config['period'], interval=config['interval'])
                
                # Add logging to debug data size
                logger.info(f"Retrieved {len(hist)} points for {tf} timeframe")
                
                if hist.empty:
                    logger.warning(f"No data received for {tf} timeframe")
                    continue
                
                if len(hist) < 30:  # Minimum required for analysis
                    logger.warning(f"Insufficient data for {tf} timeframe: {len(hist)} points")
                    continue
                    
                analyses[tf] = self._analyze_timeframe(hist, config)

            # Combine analyses for final decision
            if not analyses:
                logger.error("No valid analyses available")
                return self.get_empty_analysis()
            
            return self._combine_timeframe_analyses(analyses)

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return self.get_empty_analysis()

    def _analyze_timeframe(self, hist: pd.DataFrame, config: Dict) -> Dict:
        """Analyze single timeframe"""
        current_price = float(hist['Close'].iloc[-1])
        
        signals = {
            'current_price': current_price,
            'momentum': self._get_momentum(hist),
            'trend': self._get_trend(hist),
            'volume': self._get_volume(hist),
            'moving_averages': self._get_moving_averages(hist, config),
            'volatility': self._get_volatility(hist)
        }

        sr_levels = self._get_support_resistance(hist)
        tech_score = self._calculate_score(signals)

        return {
            'signals': signals,
            'support_resistance': sr_levels,
            'technical_score': tech_score
        }

    def _get_volatility(self, hist: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        returns = hist['Close'].pct_change()
        
        # Daily volatility
        daily_vol = returns.std()
        
        # Annualized volatility
        annual_vol = daily_vol * np.sqrt(252)
        
        # Volatility trend (increasing/decreasing)
        recent_vol = returns.tail(self.periods['volatility']).std()
        prev_vol = returns.iloc[:-self.periods['volatility']].std()
        
        vol_trend = 'increasing' if recent_vol > prev_vol * 1.1 else \
                   'decreasing' if recent_vol < prev_vol * 0.9 else \
                   'stable'
        
        return {
            'daily': float(daily_vol),
            'annual': float(annual_vol),
            'trend': vol_trend,
            'risk_level': 'high' if annual_vol > 0.4 else \
                         'medium' if annual_vol > 0.2 else \
                         'low'
        }

    def _get_trend(self, hist: pd.DataFrame) -> Dict:
        """Enhanced trend analysis"""
        try:
            logger.info(f"Starting trend analysis with {len(hist)} data points")
            
            # MACD calculation with NaN handling
            macd = ta.trend.MACD(
                hist['Close'], 
                self.periods['macd_fast'], 
                self.periods['macd_slow'], 
                self.periods['macd_signal']
            )
            macd_series = macd.macd()
            signal_series = macd.macd_signal()
            
            # Handle NaN values in MACD
            macd_series = macd_series.fillna(0)
            signal_series = signal_series.fillna(0)
            
            macd_val = float(macd_series.iloc[-1])
            signal_val = float(signal_series.iloc[-1])
            
            # ADX for trend strength - handle potential NaN values
            adx = ta.trend.ADXIndicator(hist['High'], hist['Low'], hist['Close'])
            adx_series = adx.adx()
            # Drop NaN values that ADX calculation might introduce
            adx_series = adx_series.fillna(0)
            logger.info(f"After ADX calculation and NaN handling: {len(adx_series)} valid points")
            
            adx_value = float(adx_series.iloc[-1])
            
            # Price action trend - Adapt window size based on available data
            available_points = len(hist)
            window_size = min(20, available_points - 1)  # Ensure we don't exceed available data
            logger.info(f"Using window size of {window_size} for trend analysis")
            
            closes = hist['Close'].tail(window_size)
            highs = hist['High'].tail(window_size)
            lows = hist['Low'].tail(window_size)
            
            logger.info(f"After tail selection: closes={len(closes)}, highs={len(highs)}, lows={len(lows)}")
            
            # Check if we have enough data points for trend analysis
            if len(closes) < 3:
                logger.warning("Insufficient data points for trend analysis")
                return {
                    'direction': 'neutral',
                    'strength': adx_value,
                    'macd': macd_val,
                    'macd_signal': signal_val,
                    'macd_trend': 'neutral'
                }
            
            # Calculate trends
            higher_highs = all(highs.iloc[i] <= highs.iloc[i + 1] for i in range(len(highs) - 1))
            higher_lows = all(lows.iloc[i] <= lows.iloc[i + 1] for i in range(len(lows) - 1))
            lower_highs = all(highs.iloc[i] >= highs.iloc[i + 1] for i in range(len(highs) - 1))
            lower_lows = all(lows.iloc[i] >= lows.iloc[i + 1] for i in range(len(lows) - 1))
            
            if higher_highs and higher_lows:
                pa_trend = 'strong_bullish'
            elif lower_highs and lower_lows:
                pa_trend = 'strong_bearish'
            elif higher_lows:
                pa_trend = 'bullish'
            elif lower_highs:
                pa_trend = 'bearish'
            else:
                pa_trend = 'neutral'
            
            return {
                'direction': pa_trend,
                'strength': adx_value,
                'macd': macd_val,
                'macd_signal': signal_val,
                'macd_trend': 'bullish' if macd_val > signal_val else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {
                'direction': 'neutral',
                'strength': 0,
                'macd': 0,
                'macd_signal': 0,
                'macd_trend': 'neutral'
            }

    def _combine_timeframe_analyses(self, analyses: Dict) -> Dict:
        """Combine analyses from different timeframes"""
        try:
            # Get scores for each timeframe
            timeframe_scores = {}
            for tf, analysis in analyses.items():
                timeframe_scores[tf] = {
                    'score': analysis['technical_score']['total'],
                    'signals': analysis['signals']
                }
            
            # Determine best timeframe and highest score
            best_timeframe = max(timeframe_scores.items(), key=lambda x: x[1]['score'])[0]
            highest_score = timeframe_scores[best_timeframe]['score']
            
            # Count bullish timeframes
            bullish_count = sum(
                1 for tf_data in timeframe_scores.values()
                if tf_data['signals']['trend']['direction'] in ['bullish', 'strong_bullish']
            )
            
            # Buy signal conditions - Updated for new RSI interpretation
            buy_conditions = {
                'score_threshold': highest_score >= 70,  # High overall score
                'rsi_condition': any(
                    tf_data['signals']['momentum']['rsi'] < 40  # Changed: RSI below 40 (potential oversold)
                    for tf_data in timeframe_scores.values()
                ),
                'trend_alignment': bullish_count >= 2,  # At least 2 timeframes show bullish trend
                'macd_confirmation': any(
                    tf_data['signals']['trend']['macd'] > tf_data['signals']['trend']['macd_signal']
                    for tf_data in timeframe_scores.values()
                ),
                'volume_confirmation': any(
                    tf_data['signals']['volume']['profile'] in ['high', 'increasing']
                    for tf_data in timeframe_scores.values()
                )
            }
            
            # Buy signal requires meeting at least 3 conditions
            buy_signal = sum(buy_conditions.values()) >= 3
            
            return {
                'timeframes': {
                    tf: {
                        'score': data['score'],
                        'trend': data['signals']['trend']['direction']
                    }
                    for tf, data in timeframe_scores.items()
                },
                'summary': {
                    'highest_score': highest_score,
                    'best_timeframe': best_timeframe,
                    'bullish_timeframes': bullish_count,
                    'buy_signal': buy_signal,
                    'conditions_met': sum(buy_conditions.values()),
                    'buy_conditions': buy_conditions
                },
                'signals': timeframe_scores[best_timeframe]['signals'],
                'support_resistance': analyses[best_timeframe]['support_resistance']
            }
            
        except Exception as e:
            logger.error(f"Error combining timeframe analyses: {e}")
            return self.get_empty_analysis()

    def _calculate_alignment_score(self, trends: Dict) -> float:
        """Calculate how well timeframes are aligned"""
        if not trends:
            return 0.0
            
        bullish_count = sum(1 for t in trends.values() 
                          if 'bullish' in t)
        bearish_count = sum(1 for t in trends.values() 
                          if 'bearish' in t)
        
        # Perfect alignment = 1.0, Complete disagreement = 0.0
        return max(bullish_count, bearish_count) / len(trends)

    def _get_momentum(self, hist: pd.DataFrame) -> Dict:
        """Calculate RSI and Stochastic"""
        rsi = ta.momentum.RSIIndicator(hist['Close'], self.periods['rsi']).rsi()
        stoch = ta.momentum.StochasticOscillator(hist['High'], hist['Low'], hist['Close'])
        
        return {
            'rsi': float(rsi.iloc[-1]),
            'stochastic': {
                'k': float(stoch.stoch().iloc[-1]),
                'd': float(stoch.stoch_signal().iloc[-1])
            }
        }

    def _get_volume(self, hist: pd.DataFrame) -> Dict:
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

    def _get_moving_averages(self, hist: pd.DataFrame, config: Dict) -> Dict:
        """Calculate moving averages"""
        ma_short = ta.trend.SMAIndicator(hist['Close'], self.periods['ma_short']).sma_indicator()
        ma_long = ta.trend.SMAIndicator(hist['Close'], self.periods['ma_long']).sma_indicator()
        
        current_short = float(ma_short.iloc[-1])
        current_long = float(ma_long.iloc[-1])
        
        return {
            'trend': 'bullish' if current_short > current_long else 'bearish',
            'aligned': current_short > current_long
        }

    def _get_support_resistance(self, hist: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        price_data = hist['Close'].values
        window = 20
        
        support = sorted(price_data[argrelextrema(price_data, np.less, order=window)[0]], reverse=True)
        resistance = sorted(price_data[argrelextrema(price_data, np.greater, order=window)[0]])
        
        return {
            'support': support[:3],
            'resistance': resistance[:3]
        }

    def _calculate_score(self, signals: Dict) -> Dict:
        """Calculate technical score with more granular analysis"""
        # Momentum score (RSI based)
        rsi = signals['momentum']['rsi']
        stoch_k = signals['momentum']['stochastic']['k']
        
        # More granular RSI scoring - Fixed to align with traditional interpretation
        momentum_score = (
            90 if rsi < 30 else         # Strong buy (oversold)
            80 if 30 <= rsi < 35 else   # Buy zone
            70 if 35 <= rsi < 40 else   # Weak buy
            60 if 40 <= rsi < 45 else   # Slightly bullish
            50 if 45 <= rsi < 55 else   # Neutral
            40 if 55 <= rsi < 60 else   # Slightly bearish
            30 if 60 <= rsi < 65 else   # Weak sell
            20 if 65 <= rsi < 70 else   # Sell zone
            10                          # Strong sell (overbought >70)
        )
        
        # Add stochastic influence - also aligned with traditional interpretation
        stoch_score = 100 - abs(stoch_k - 50)  # Higher score when closer to extremes
        momentum_score = (momentum_score + stoch_score) / 2

        # Trend score with MACD influence
        base_trend_score = {
            'strong_bullish': 100,
            'bullish': 75,
            'neutral': 50,
            'bearish': 25,
            'strong_bearish': 0
        }[signals['trend']['direction']]
        
        # Adjust trend score based on MACD
        macd_diff = signals['trend']['macd'] - signals['trend']['macd_signal']
        trend_score = base_trend_score + (macd_diff * 10)
        trend_score = max(0, min(100, trend_score))

        # Volume score with volatility influence
        base_volume_score = {
            'high': 100,
            'increasing': 75,
            'normal': 50,
            'decreasing': 25
        }[signals['volume']['profile']]
        
        # Adjust volume score based on volatility risk
        vol_adjustment = {
            'low': 10,
            'medium': 0,
            'high': -10
        }[signals['volatility']['risk_level']]
        
        volume_score = base_volume_score + vol_adjustment
        volume_score = max(0, min(100, volume_score))

        # Final weighted score with moving average alignment bonus
        total_score = (
            momentum_score * 0.3 +    # 30% weight
            trend_score * 0.5 +       # 50% weight
            volume_score * 0.2        # 20% weight
        )
        
        # Add bonus for moving average alignment
        if signals['moving_averages']['aligned']:
            total_score += 5
        
        total_score = max(0, min(100, total_score))

        return {
            'total': total_score,
            'momentum': momentum_score,
            'trend': trend_score,
            'volume': volume_score
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
            'technical_score': {'total': 50, 'momentum': 50, 'trend': 50, 'volume': 50},
            'timeframes': {
                'short': {'trend': 'neutral', 'score': 50},
                'medium': {'trend': 'neutral', 'score': 50},
                'long': {'trend': 'neutral', 'score': 50}
            }
        }

def run_test_analysis():
    """Test function to verify technical analysis"""
    logger.info("Starting technical analysis test...")
    
    test_ticker = "AAPL"
    analyzer = TechnicalAnalyzer(timeframe='medium')
    
    try:
        # Get stock data first to verify we have enough data
        stock = yf.Ticker(test_ticker)
        hist = stock.history(period="3mo", interval="1d")  # Get 3 months of daily data
        
        if len(hist) < 30:
            logger.error(f"Insufficient data points. Got {len(hist)}, need at least 30")
            return False
            
        logger.info(f"Retrieved {len(hist)} data points for analysis")
        
        analysis = analyzer.analyze_stock(test_ticker)
        
        print("\n=== Technical Analysis Test Results ===")
        print(f"Stock: {test_ticker}")
        
        # First show the main (medium term) analysis
        print("\n=== MEDIUM TERM ANALYSIS (Primary) ===")
        print("\n1. Price Information:")
        print(f"Current Price: ${analysis['signals']['current_price']:.2f}")
        
        print("\n2. Momentum Analysis:")
        print(f"RSI: {analysis['signals']['momentum']['rsi']:.2f}")
        print(f"Stochastic K: {analysis['signals']['momentum']['stochastic']['k']:.2f}")
        print(f"Stochastic D: {analysis['signals']['momentum']['stochastic']['d']:.2f}")
        
        print("\n3. Trend Analysis:")
        print(f"Direction: {analysis['signals']['trend']['direction']}")
        print(f"Strength (ADX): {analysis['signals']['trend']['strength']:.2f}")
        print(f"MACD: {analysis['signals']['trend']['macd']:.2f}")
        print(f"MACD Signal: {analysis['signals']['trend']['macd_signal']:.2f}")
        print(f"MACD Trend: {analysis['signals']['trend']['macd_trend']}")
        
        print("\n4. Volume Profile:")
        print(f"Status: {analysis['signals']['volume']['profile']}")
        print(f"Value: {analysis['signals']['volume']['value']:,.0f}")
        
        print("\n5. Moving Averages:")
        print(f"Short MA Period: {analyzer.config['ma_short']} days")
        print(f"Long MA Period: {analyzer.config['ma_long']} days")
        print(f"Trend: {analysis['signals']['moving_averages']['trend']}")
        print(f"Aligned: {analysis['signals']['moving_averages']['aligned']}")
        
        print("\n6. Volatility Analysis:")
        print(f"Daily: {analysis['signals']['volatility']['daily']:.4f}")
        print(f"Annual: {analysis['signals']['volatility']['annual']:.4f}")
        print(f"Trend: {analysis['signals']['volatility']['trend']}")
        print(f"Risk Level: {analysis['signals']['volatility']['risk_level']}")
        
        print("\n7. Support/Resistance Levels:")
        print(f"Support Levels: {[f'${x:.2f}' for x in analysis['support_resistance']['support']]}")
        print(f"Resistance Levels: {[f'${x:.2f}' for x in analysis['support_resistance']['resistance']]}")
        
        print("\n8. Technical Score:")
        print(f"Total Score: {analysis['technical_score']['total']:.2f}")
        print(f"Momentum Score: {analysis['technical_score']['momentum']:.2f}")
        print(f"Trend Score: {analysis['technical_score']['trend']:.2f}")
        print(f"Volume Score: {analysis['technical_score']['volume']:.2f}")
        
        # Now show comparison across timeframes
        print("\n=== MULTI-TIMEFRAME COMPARISON ===")
        for tf, data in analysis['timeframes'].items():
            print(f"\n{tf.upper()} TERM:")
            print(f"Period: {TIMEFRAME_CONFIGS[tf]['period']}")
            print(f"MA Periods: {TIMEFRAME_CONFIGS[tf]['ma_short']}/{TIMEFRAME_CONFIGS[tf]['ma_long']} days")
            print(f"Trend Direction: {data['trend']}")
            print(f"Technical Score: {data['score']:.2f}")
            
        # Show timeframe alignment
        bullish_timeframes = sum(1 for data in analysis['timeframes'].values() if 'bullish' in data['trend'])
        bearish_timeframes = sum(1 for data in analysis['timeframes'].values() if 'bearish' in data['trend'])
        neutral_timeframes = sum(1 for data in analysis['timeframes'].values() if data['trend'] == 'neutral')
        
        print("\nTimeframe Alignment:")
        print(f"Bullish Timeframes: {bullish_timeframes}")
        print(f"Bearish Timeframes: {bearish_timeframes}")
        print(f"Neutral Timeframes: {neutral_timeframes}")
        
        # Final Analysis Summary
        print("\n=== FINAL ANALYSIS SUMMARY ===")
        avg_score = sum(data['score'] for data in analysis['timeframes'].values()) / len(analysis['timeframes'])
        print(f"Average Score Across Timeframes: {avg_score:.2f}")
        print(f"Primary Timeframe Score: {analysis['technical_score']['total']:.2f}")
        print(f"Overall Trend Alignment: {'Strong' if bullish_timeframes == 3 or bearish_timeframes == 3 else 'Mixed'}")
        print(f"Risk Level: {analysis['signals']['volatility']['risk_level'].upper()}")
        
        # Add to test function, before "=== Test Complete ==="
        print("\n=== TRADING SIGNAL ANALYSIS ===")
        print(f"Highest Score: {analysis['summary']['highest_score']:.2f}")
        print(f"Best Timeframe: {analysis['summary']['best_timeframe'].upper()}")
        print(f"Timeframe Alignment: {analysis['summary']['alignment_score']:.2f}")
        print(f"Valid Setup: {'YES' if analysis['summary']['is_valid_setup'] else 'NO'}")
        print(f"Buy Signal: {'YES' if analysis['summary']['buy_signal'] else 'NO'}")
        if analysis['summary']['buy_signal']:
            print("\nRECOMMENDATION: BUY")
            print(f"Best Entry Timeframe: {analysis['summary']['best_timeframe'].upper()}")
            print(f"Current Price: ${analysis['signals']['current_price']:.2f}")
            support_levels = analysis['support_resistance']['support']
            print(f"Stop Loss Suggestion: ${support_levels[0]:.2f}")
        
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