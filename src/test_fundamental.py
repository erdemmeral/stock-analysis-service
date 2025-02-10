from fundamental_analysis import FundamentalAnalyzer
import json

def test_fundamental():
    analyzer = FundamentalAnalyzer()
    results, raw_data = analyzer.analyze_stocks()
    
    # Save results
    with open('fundamental_analysis_results.json', 'w') as f:
        json.dump({
            'passing_results': results,
            'all_data': raw_data
        }, f, indent=4)
    
    print(f"\nFundamental Analysis Complete!")
    print(f"Found {len(results)} stocks passing criteria")
    
    # Print top stocks
    print("\nTop Fundamental Stocks:")
    print("-" * 60)
    print(f"{'Ticker':<8} {'Score':<8} {'D/E':<8} {'Net Margin':<12} {'EPS Growth'}")
    print("-" * 60)
    
    for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:10]:
        metrics = stock['metrics']
        print(f"{stock['ticker']:<8} "
              f"{stock['score']:>6.1f}% "
              f"{metrics.get('debt_to_equity', 'N/A'):>7.2f} "
              f"{metrics.get('net_margin', 'N/A'):>10.1f}% "
              f"{metrics.get('eps_growth_5y', 'N/A'):>9.1f}%")

if __name__ == "__main__":
    test_fundamental() 