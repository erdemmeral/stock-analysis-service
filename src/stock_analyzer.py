import pandas as pd
from fundamental_analysis import FundamentalAnalyzer
from tqdm import tqdm
import json
from datetime import datetime

class StockAnalyzer:
    def __init__(self):
        self.fundamental_analyzer = FundamentalAnalyzer()
        
    def analyze_stocks(self, tickers):
        """
        Analyzes a list of stocks and returns their fundamental scores
        """
        results = []
        raw_data = []
        
        # Show progress bar
        for ticker in tqdm(tickers, desc="Analyzing stocks"):
            try:
                data = self.fundamental_analyzer.get_fundamental_data(ticker)
                if data is None:  # No gross profit data
                    raw_data.append({
                        'ticker': ticker,
                        'status': 'missing_gross_profit',
                        'meets_criteria': False
                    })
                    continue
                
                # Check if we have all required metrics
                if 'missing_metrics' in data:
                    raw_data.append({
                        'ticker': ticker,
                        'status': 'missing_data',
                        'missing_metrics': data['missing_metrics'],
                        'raw_metrics': data
                    })
                    continue
                
                score = self.fundamental_analyzer.score_fundamentals(data)
                
                raw_data.append({
                    'ticker': ticker,
                    'status': 'analyzed',
                    'score': score,
                    'raw_metrics': {
                        # Basic metrics
                        'debt_to_equity': data.get('debt_to_equity', 'N/A'),
                        'eps_growth_5y': data.get('eps_growth_5y', 'N/A'),
                        'gross_margin': data.get('gross_margin', 'N/A'),
                        'net_margin': data.get('net_margin', 'N/A'),
                        'operating_margin': data.get('operating_margin', 'N/A'),
                        'pe_ratio': data.get('pe_ratio', 'N/A'),
                        'country': data.get('country', 'N/A'),
                        
                        # Buffett metrics
                        'gross_profit': data.get('gross_profit', 'N/A'),
                        'operating_income': data.get('operating_income', 'N/A'),
                        'net_income': data.get('net_income', 'N/A'),
                        'sga': data.get('sga', 'N/A'),
                        'depreciation': data.get('depreciation', 'N/A'),
                        'interest_expense': data.get('interest_expense', 'N/A'),
                        'long_term_debt': data.get('long_term_debt', 'N/A'),
                        'total_liabilities': data.get('total_liabilities', 'N/A'),
                        'shareholders_equity': data.get('shareholders_equity', 'N/A'),
                        
                        # Calculated ratios
                        'sga_to_gross_profit': data.get('sga_to_gross_profit', 'N/A'),
                        'depreciation_to_gross_profit': data.get('depreciation_to_gross_profit', 'N/A'),
                        'interest_to_operating_income': data.get('interest_to_operating_income', 'N/A'),
                        'debt_coverage': data.get('debt_coverage', 'N/A'),
                        'leverage_ratio': data.get('leverage_ratio', 'N/A')
                    },
                    'meets_criteria': score > 0
                })
                
                if score > 0:
                    results.append({
                        'ticker': ticker,
                        'score': score,
                        'metrics': data
                    })
            except Exception as e:
                raw_data.append({
                    'ticker': ticker,
                    'status': 'error',
                    'error': str(e),
                    'meets_criteria': False
                })
                print(f"Error analyzing {ticker}: {str(e)}")
                continue
        
        # Print summary
        total = len(tickers)
        analyzed = sum(1 for x in raw_data if x['status'] == 'analyzed')
        missing_data = sum(1 for x in raw_data if x['status'] == 'missing_data')
        errors = sum(1 for x in raw_data if x['status'] == 'error')
        passing = len(results)
        
        print(f"\nAnalysis Summary:")
        print(f"Total stocks: {total}")
        print(f"Successfully analyzed: {analyzed}")
        print(f"Missing critical data: {missing_data}")
        print(f"Errors: {errors}")
        print(f"Passing all criteria: {passing}")
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results, raw_data
    
    def save_results(self, results, raw_data, filename):
        """
        Saves analysis results to a JSON file
        """
        output = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'passing_results': results,
            'all_data': raw_data,
            'analysis_parameters': {
                'fundamental_metrics': self.fundamental_analyzer.fundamental_metrics,
                'buffett_metrics': self.fundamental_analyzer.buffett_metrics
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=4)
    
    def print_results(self, results):
        """
        Prints analysis results in a formatted way
        """
        print("\nTop Stocks by Fundamental Score:")
        print("-" * 80)
        print(f"{'Ticker':<8} {'Score':<8} {'D/E':<8} {'Net Margin':<12} {'Gross Margin':<12} {'EPS Growth':<10}")
        print("-" * 80)
        
        for result in results:  # Show top 20 stocks
            metrics = result['metrics']
            print(f"{result['ticker']:<8} "
                  f"{result['score']:>6.1f}% "
                  f"{metrics.get('debt_to_equity', 'N/A'):>7.2f} "
                  f"{metrics.get('net_margin', 'N/A'):>10.1f}% "
                  f"{metrics.get('gross_margin', 'N/A'):>11.1f}% "
                  f"{metrics.get('eps_growth_5y', 'N/A'):>9.1f}%")

def main():
    # Read tickers from file
    with open('stock_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    # Create analyzer and run analysis
    analyzer = StockAnalyzer()
    print(f"Analyzing {len(tickers)} stocks...")
    
    # Get minimum score for reference
    min_score = analyzer.fundamental_analyzer.calculate_minimum_score()
    print(f"\nMinimum passing score: {min_score:.2f}%")
    
    # Run analysis
    results, raw_data = analyzer.analyze_stocks(tickers)
    
    # Save results
    analyzer.save_results(results, raw_data, 'fundamental_analysis_results.json')
    
    # Print results
    analyzer.print_results(results)
    
    print(f"\nAnalysis complete. Found {len(results)} stocks meeting criteria.")
    print(f"Full results (including raw data) saved to fundamental_analysis_results.json")

if __name__ == "__main__":
    main() 