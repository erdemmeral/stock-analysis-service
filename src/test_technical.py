import unittest
import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalyzer

class TestTechnicalAnalyzer(unittest.TestCase):
    def setUp(self):
        """Setup test data"""
        self.analyzer = TechnicalAnalyzer()
        
        # Create test data with known patterns
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        n_days = len(dates)
        
        # Create base data with realistic price movements
        base_price = 100
        prices = []
        volumes = []
        
        # Create a pattern with clear trends for MACD
        for i in range(n_days):
            if i < n_days // 4:  # Strong uptrend
                base_price *= 1.003
                volume = np.random.uniform(1500000, 2000000)
            elif i < n_days // 3:  # First top
                base_price *= 0.999
                volume = np.random.uniform(2000000, 2500000)
            elif i < n_days // 2:  # Sharp downtrend
                base_price *= 0.997
                volume = np.random.uniform(2500000, 3000000)
            elif i < 2 * n_days // 3:  # Recovery
                base_price *= 1.002
                volume = np.random.uniform(2000000, 2500000)
            else:  # Sideways with slight downtrend
                base_price *= 0.9995
                volume = np.random.uniform(1500000, 2000000)
            
            # Add minimal noise to keep trends clear
            base_price *= (1 + np.random.normal(0, 0.0005))
            prices.append(base_price)
            volumes.append(volume)
        
        # Create DataFrame with realistic OHLCV data
        self.test_data = pd.DataFrame({
            'Open': prices,
            'Close': [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'Volume': volumes
        }, index=dates)

    def test_setup_indicators(self):
        """Test indicator setup and parameters"""
        self.assertIn('medium', self.analyzer.indicators)
        self.assertIn('long', self.analyzer.indicators)
        
        medium = self.analyzer.indicators['medium']
        self.assertEqual(medium['ma_short'], 20)
        self.assertEqual(medium['ma_long'], 50)
        self.assertEqual(medium['rsi_period'], 14)
        
        long = self.analyzer.indicators['long']
        self.assertEqual(long['ma_short'], 50)
        self.assertEqual(long['ma_long'], 200)
        self.assertEqual(long['fibonacci_period'], 252)

    def test_calculate_indicators(self):
        """Test technical indicator calculations"""
        signals = self.analyzer.calculate_indicators(self.test_data)
        
        # Check structure
        self.assertIn('indicators', signals)
        indicators = signals['indicators']
        
        # Check moving averages
        self.assertIn('moving_averages', indicators)
        ma = indicators['moving_averages']
        self.assertIn('ma_trend', ma)
        self.assertIn('ma_short', ma)
        self.assertIn('ma_long', ma)
        self.assertIn(ma['ma_trend'], ['bullish', 'bearish'])
        
        # Check momentum
        self.assertIn('momentum', indicators)
        momentum = indicators['momentum']
        self.assertIn('rsi', momentum)
        self.assertIn('rsi_signal', momentum)
        self.assertTrue(0 <= momentum['rsi'] <= 100)
        self.assertIn(momentum['rsi_signal'], ['overbought', 'oversold', 'bullish', 'bearish'])
        
        # Check trend
        self.assertIn('trend', indicators)
        trend = indicators['trend']
        self.assertIn('macd', trend)
        self.assertIn('macd_signal', trend)
        self.assertIn('macd_trend', trend)
        self.assertIn(trend['macd_trend'], ['bullish', 'bearish'])
        
        # Check volume
        self.assertIn('volume', indicators)
        volume = indicators['volume']
        self.assertIn('mfi', volume)
        self.assertIn('obv_trend', volume)
        self.assertTrue(0 <= volume['mfi'] <= 100)
        self.assertIn(volume['obv_trend'], ['up', 'down', 'sideways', 'unknown'])

    def test_pattern_recognition(self):
        """Test pattern recognition methods"""
        patterns = self.analyzer._check_patterns(self.test_data)
        
        expected_patterns = [
            'double_top', 'double_bottom', 'head_shoulders',
            'inverse_head_shoulders', 'bullish_flag', 'bearish_flag',
            'ascending_triangle', 'descending_triangle'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns)
            self.assertIn(patterns[pattern], (True, False))

    def test_support_resistance(self):
        """Test support and resistance calculations"""
        levels = self.analyzer.find_support_resistance(self.test_data)
        
        self.assertIn('support', levels)
        self.assertIn('resistance', levels)
        self.assertIn('current_price', levels)
        
        support = levels['support']
        resistance = levels['resistance']
        
        self.assertIn('levels', support)
        self.assertIn('strength', support)
        self.assertIn('levels', resistance)
        self.assertIn('strength', resistance)
        
        # Check level values
        if support['levels']:
            self.assertTrue(all(s < levels['current_price'] for s in support['levels']))
        if resistance['levels']:
            self.assertTrue(all(r > levels['current_price'] for r in resistance['levels']))

    def test_trend_analysis(self):
        """Test trend analysis"""
        trend = self.analyzer.analyze_trend(self.test_data)
        
        self.assertIn('trend', trend)
        self.assertIn('strength', trend)
        self.assertIn('price_above_ma', trend)
        self.assertIn('ma_alignment', trend)
        self.assertIn('momentum', trend)
        
        self.assertIn(trend['trend'], ['strong_uptrend', 'uptrend', 'strong_downtrend', 'downtrend', 'sideways'])
        self.assertTrue(0 <= trend['strength'] <= 100)
        self.assertIsInstance(trend['price_above_ma'], bool)
        self.assertIsInstance(trend['ma_alignment'], bool)
        self.assertIn(trend['momentum'], ['increasing', 'decreasing', 'unknown'])

    def test_fibonacci_levels(self):
        """Test Fibonacci level calculations"""
        fib_levels = self.analyzer.calculate_fibonacci_levels(self.test_data)
        
        expected_levels = ['level_0', 'level_0.236', 'level_0.382', 
                         'level_0.5', 'level_0.618', 'level_1']
        
        for level in expected_levels:
            self.assertIn(level, fib_levels)
            self.assertIsInstance(fib_levels[level], float)
        
        # Check levels are in ascending order
        levels = [fib_levels[f'level_{l}'] for l in ['0', '0.236', '0.382', '0.5', '0.618', '1']]
        self.assertEqual(levels, sorted(levels))

    def test_predict_price_movement(self):
        """Test ensemble prediction method"""
        prediction = self.analyzer.predict_price_movement(self.test_data)
        
        # Check basic structure
        self.assertIsNotNone(prediction)
        self.assertIn('trend', prediction)
        self.assertIn('market_condition', prediction)
        
        # Check trend
        trend = prediction['trend']
        self.assertIn('direction', trend)
        self.assertIn('predicted_change_percent', trend)
        self.assertIn('confidence', trend)
        self.assertIn(trend['direction'], ['bullish', 'bearish'])
        self.assertIsInstance(trend['predicted_change_percent'], float)
        self.assertTrue(0 <= trend['confidence'] <= 1)
        
        # Check market condition
        market_condition = prediction['market_condition']
        self.assertIn('volatility', market_condition)
        self.assertIn('trend_strength', market_condition)
        self.assertIsInstance(market_condition['volatility'], float)
        self.assertTrue(0 <= market_condition['volatility'] <= 1, 
                       f"Volatility {market_condition['volatility']} out of range")
        
        # Check predictions if they exist
        if 'predictions' in prediction:
            predictions = prediction['predictions']
            self.assertIsNotNone(predictions, "Predictions should not be None")
            
            expected_methods = ['trend_following', 'momentum', 'pattern', 'support_resistance']
            for method in expected_methods:
                self.assertIn(method, predictions)
                self.assertIsInstance(predictions[method], float)
                self.assertTrue(-10 <= predictions[method] <= 10, 
                              f"{method} prediction {predictions[method]} out of range")

    def test_momentum_prediction(self):
        """Test momentum prediction separately"""
        signals = self.analyzer.calculate_indicators(self.test_data)
        
        # Verify signal structure
        self.assertIn('indicators', signals)
        indicators = signals['indicators']
        
        # Check momentum indicators
        self.assertIn('momentum', indicators)
        momentum = indicators['momentum']
        self.assertIn('rsi', momentum)
        self.assertIn('rsi_signal', momentum)
        
        # Check trend indicators (where MACD actually is)
        self.assertIn('trend', indicators)
        trend = indicators['trend']
        self.assertIn('macd', trend)
        self.assertIn('macd_signal', trend)
        self.assertIn('macd_trend', trend)
        
        # Test momentum prediction
        prediction = self.analyzer._momentum_prediction(signals)
        self.assertIsInstance(prediction, float)
        self.assertTrue(-2 <= prediction <= 2, 
                       f"Momentum prediction {prediction} out of range")

def main():
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    main() 