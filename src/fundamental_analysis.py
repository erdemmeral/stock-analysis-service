from typing import Dict, List
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import time
import logging
import os

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    def __init__(self):
        # Basic screening metrics with minimum and maximum thresholds
        self.fundamental_metrics = {
            'debt_to_equity': {'min': 0, 'max': 1.0, 'weight': 1.0},
            'eps_growth_5y': {'min': 0, 'required_min': -10, 'weight': 1.0},  # Allow some negative growth but penalize
            'gross_margin': {'min': 30, 'required_min': 20, 'weight': 1.0},   # Minimum 20% required
            'net_margin': {'min': 15, 'required_min': 10, 'weight': 1.0},     # Minimum 10% required
            'operating_margin': {'min': 0, 'required_min': -5, 'weight': 1.0}, # Allow slight negative
            'pe_ratio': {'min': 0, 'max': 50, 'weight': 1.0},                 # Add maximum PE
            'country': {'value': 'USA', 'weight': 1.0}
        }
        
        # Buffett criteria with minimum and maximum thresholds
        self.buffett_metrics = {
            'sga_to_gross_profit': {'max': 0.30, 'required_max': 0.40, 'weight': 1.0},  # Max 40% required
            'depreciation_to_gross_profit': {'max': 0.10, 'required_max': 0.15, 'weight': 1.0},  # Max 15% required
            'interest_to_operating_income': {'max': 0.15, 'required_max': 0.20, 'weight': 1.0},  # Max 20% required
            'debt_coverage': {'min': 4.0, 'required_min': 3.0, 'weight': 1.0},  # Minimum 3x coverage required
            'leverage_ratio': {'max': 1.0, 'required_max': 1.2, 'weight': 1.0}   # Max 1.2x leverage required
        }
    
    def get_fundamental_data(self, ticker: str) -> Dict:
        """
        Fetches fundamental data for a given stock using yfinance
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            
            # Try to get gross profit first - if not available, return None
            gross_profit = None
            gross_profit_names = ['Gross Profit', 'GrossProfit']
            for name in gross_profit_names:
                try:
                    if name in income_stmt.index:
                        gross_profit = income_stmt.loc[name].iloc[0]
                        break
                except:
                    continue
            
            # If no gross profit found, try calculating it
            if gross_profit is None:
                try:
                    revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    cogs = income_stmt.loc['Cost Of Revenue'].iloc[0]
                    gross_profit = revenue - cogs
                except:
                    print(f"{ticker}: No gross profit data available")
                    return None  # Skip this stock if no gross profit data
            
            # Basic metrics with validation
            basic_metrics = {}
            
            # Check each critical metric
            critical_metrics = {
                'debt_to_equity': ('debtToEquity', float('inf')),
                'eps_growth_5y': ('earningsGrowth', None),  # Changed from float('-inf')
                'gross_margin': ('grossMargins', None),
                'net_margin': ('profitMargins', None),
                'operating_margin': ('operatingMargins', None),
                'pe_ratio': ('trailingPE', None),
                'country': ('country', 'Unknown')
            }
            
            missing_metrics = []
            for metric, (info_key, default) in critical_metrics.items():
                value = info.get(info_key, default)
                if value is None or value in [float('inf'), float('-inf')]:
                    missing_metrics.append(metric)
                else:
                    if metric != 'country':
                        value = value * 100 if metric == 'eps_growth_5y' else value
                    basic_metrics[metric] = value
            
            if missing_metrics:
                print(f"{ticker}: Missing critical metrics: {', '.join(missing_metrics)}")
                basic_metrics['missing_metrics'] = missing_metrics
            
            # Store all raw values
            data = {
                # Basic metrics from info
                'debt_to_equity': info.get('debtToEquity', float('inf')),
                'eps_growth_5y': info.get('earningsGrowth', float('-inf')) * 100,
                'gross_margin': info.get('grossMargins', float('-inf')) * 100,
                'net_margin': info.get('profitMargins', float('-inf')) * 100,
                'operating_margin': info.get('operatingMargins', float('-inf')) * 100,
                'pe_ratio': info.get('trailingPE', float('-inf')),
                'country': info.get('country', 'Unknown')
            }
            
            # Try to get all raw values from financial statements
            try:
                # Store raw values before calculations
                data.update(self._get_raw_financial_data(income_stmt, balance_sheet))
                
                # Calculate ratios if we have the required raw values
                if all(k in data for k in ['gross_profit', 'sga', 'operating_income', 'net_income', 'long_term_debt']):
                    data.update({
                        'sga_to_gross_profit': data['sga'] / data['gross_profit'] if data['gross_profit'] else float('inf'),
                        'depreciation_to_gross_profit': data['depreciation'] / data['gross_profit'] if data['gross_profit'] else float('inf'),
                        'interest_to_operating_income': data['interest_expense'] / data['operating_income'] if data['operating_income'] else float('inf'),
                        'debt_coverage': data['net_income'] * 4 / data['long_term_debt'] if data['long_term_debt'] else float('inf'),
                        'leverage_ratio': data['total_liabilities'] / data['shareholders_equity'] if data['shareholders_equity'] else float('inf')
                    })
                
            except Exception as e:
                print(f"Error calculating ratios for {ticker}: {str(e)}")
            
            return data
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def _get_raw_financial_data(self, income_stmt, balance_sheet):
        """Helper method to get raw financial data"""
        data = {}
        
        # Try different possible names for each metric
        metrics_map = {
            'gross_profit': ['Gross Profit', 'GrossProfit', 'Total Revenue', 'TotalRevenue'],
            'operating_income': ['Operating Income', 'OperatingIncome', 'EBIT'],
            'net_income': ['Net Income', 'NetIncome', 'Net Income Common Stockholders'],
            'sga': ['Selling General And Administration', 'SGA Expense', 'Operating Expenses'],
            'depreciation': ['Reconciled Depreciation', 'Depreciation And Amortization In Income Statement', 'Depreciation'],
            'interest_expense': ['Interest Expense', 'Other Income Expense'],
            'long_term_debt': ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation', 'Total Non Current Liabilities Net Minority Interest'],
            'total_liabilities': ['Total Liabilities Net Minority Interest'],
            'shareholders_equity': ['Stockholders Equity']
        }
        
        for metric, possible_names in metrics_map.items():
            for name in possible_names:
                try:
                    if name in income_stmt.index:
                        data[metric] = income_stmt.loc[name].iloc[0]
                        break
                    elif name in balance_sheet.index:
                        data[metric] = balance_sheet.loc[name].iloc[0]
                        break
                except:
                    continue
        
        # Try to calculate gross profit if not found
        if 'gross_profit' not in data:
            try:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                cogs = income_stmt.loc['Cost Of Revenue'].iloc[0]
                data['gross_profit'] = revenue - cogs
            except:
                pass
        
        return data
    
    def score_fundamentals(self, fundamental_data: Dict) -> float:
        """
        Scores the stock based on both fundamental and Buffett metrics
        Returns a score between 0 and 100, or 0 if minimum criteria not met
        """
        if fundamental_data is None:
            return 0
        
        # First check minimum requirements
        if (fundamental_data.get('debt_to_equity', float('inf')) > 1.0 or
            fundamental_data.get('eps_growth_5y', float('-inf')) < 0 or
            fundamental_data.get('gross_margin', 0) < 30 or
            fundamental_data.get('net_margin', 0) < 15 or
            fundamental_data.get('operating_margin', float('-inf')) < 0):
            return 0
        
        score = 0
        total_weight = 0
        
        # Score basic screening metrics
        for metric, criteria in self.fundamental_metrics.items():
            if metric == 'country':
                if fundamental_data[metric] == criteria['value']:
                    score += criteria['weight']
                total_weight += criteria['weight']
                continue
                
            value = fundamental_data[metric]
            weight = criteria['weight']
            total_weight += weight
            
            # Skip scoring if value is invalid
            if value is None or value in [float('inf'), float('-inf')] or pd.isna(value):
                continue
            
            if 'max' in criteria and 'min' in criteria:
                # For metrics with both min and max
                if value >= criteria['min'] and value <= criteria['max']:
                    score += weight
                elif value < criteria['min'] and criteria['min'] != 0:
                    score += weight * (value / criteria['min'])
                elif value > criteria['max']:
                    score += weight * (criteria['max'] / value)
            elif 'min' in criteria:
                # For metrics with only minimum
                if criteria['min'] == 0:
                    score += weight if value >= 0 else 0
                else:
                    if value >= criteria['min']:
                        score += weight
                    else:
                        score += weight * (value / criteria['min'])
            elif 'max' in criteria:
                # For metrics with only maximum
                if value <= criteria['max']:
                    score += weight
                else:
                    score += weight * (criteria['max'] / value)
        
        # Score Buffett metrics if available
        if 'sga_to_gross_profit' in fundamental_data:
            for metric, criteria in self.buffett_metrics.items():
                value = fundamental_data[metric]
                weight = criteria['weight']
                total_weight += weight
                
                # Skip scoring if value is invalid
                if value is None or value in [float('inf'), float('-inf')] or pd.isna(value):
                    continue
                
                if 'min' in criteria:
                    if criteria['min'] == 0:
                        score += weight if value >= 0 else 0
                    else:
                        if value >= criteria['min']:
                            score += weight
                        else:
                            score += weight * (value / criteria['min'])
                elif 'max' in criteria:
                    if value <= criteria['max']:
                        score += weight
                    else:
                        score += weight * (criteria['max'] / value)
        
        # Convert score to percentage
        return (score / total_weight) * 100 if total_weight > 0 else 0

    def meets_minimum_criteria(self, data: Dict) -> bool:
        """
        Checks if stock meets minimum required criteria
        """
        # Check basic screening criteria
        for metric, criteria in self.fundamental_metrics.items():
            if metric == 'country':
                if data[metric] != criteria['value']:
                    return False
                continue
                
            value = data[metric]
            if 'required_min' in criteria and value < criteria['required_min']:
                return False
            if 'required_max' in criteria and value > criteria['required_max']:
                return False
        
        # Check Buffett criteria if available
        if 'sga_to_gross_profit' in data:
            for metric, criteria in self.buffett_metrics.items():
                value = data[metric]
                if 'required_min' in criteria and value < criteria['required_min']:
                    return False
                if 'required_max' in criteria and value > criteria['required_max']:
                    return False
        
        return True
    
    def meets_all_criteria(self, ticker: str) -> bool:
        """
        Returns True if stock meets all fundamental and Buffett criteria
        """
        data = self.get_fundamental_data(ticker)
        if data is None:
            return False
            
        # Check basic screening criteria
        basic_criteria = (
            data['debt_to_equity'] <= self.fundamental_metrics['debt_to_equity']['max'] and
            data['eps_growth_5y'] >= self.fundamental_metrics['eps_growth_5y']['min'] and
            data['gross_margin'] >= self.fundamental_metrics['gross_margin']['min'] and
            data['net_margin'] >= self.fundamental_metrics['net_margin']['min'] and
            data['operating_margin'] >= self.fundamental_metrics['operating_margin']['min'] and
            data['pe_ratio'] >= self.fundamental_metrics['pe_ratio']['min'] and
            data['country'] == self.fundamental_metrics['country']['value']
        )
        
        # Check Buffett criteria if available
        if 'sga_to_gross_profit' in data:
            buffett_criteria = (
                data['sga_to_gross_profit'] <= self.buffett_metrics['sga_to_gross_profit']['max'] and
                data['depreciation_to_gross_profit'] <= self.buffett_metrics['depreciation_to_gross_profit']['max'] and
                data['interest_to_operating_income'] <= self.buffett_metrics['interest_to_operating_income']['max'] and
                data['debt_coverage'] >= self.buffett_metrics['debt_coverage']['min'] and
                data['leverage_ratio'] <= self.buffett_metrics['leverage_ratio']['max']
            )
            return basic_criteria and buffett_criteria
            
        return basic_criteria 

    def calculate_minimum_score(self):
        """
        Calculates the score a stock would get if it just meets minimum requirements
        """
        minimum_data = {
            # Basic metrics at minimum levels
            'debt_to_equity': 1.0,
            'eps_growth_5y': -10,
            'gross_margin': 20,
            'net_margin': 10,
            'operating_margin': -5,
            'pe_ratio': 50,
            'country': 'USA',
            
            # Buffett metrics at minimum levels
            'sga_to_gross_profit': 0.40,
            'depreciation_to_gross_profit': 0.15,
            'interest_to_operating_income': 0.20,
            'debt_coverage': 3.0,
            'leverage_ratio': 1.2
        }
        
        return self.score_fundamentals(minimum_data) 

    def analyze_stocks(self):
        """Analyze stocks from stock_tickers.txt"""
        try:
            # Check if file exists
            if not os.path.exists('stock_tickers.txt'):
                logger.error("stock_tickers.txt not found!")
                return [], []

            # Read tickers from file
            with open('stock_tickers.txt', 'r') as f:
                tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Found {len(tickers)} tickers in stock_tickers.txt")
            
            results = []
            raw_data = []
            
            # Process in batches
            batch_size = 30
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.info(f"Processing batch {(i//batch_size)+1} of {(len(tickers)+batch_size-1)//batch_size}")
                
                for ticker in tqdm(batch, desc="Analyzing stocks"):
                    try:
                        logger.info(f"Analyzing fundamentals for {ticker}")
                        analysis = self.analyze_stock(ticker)
                        logger.info(f"{ticker} analysis result: {analysis['status']}")
                        
                        if analysis['meets_criteria']:
                            logger.info(f"{ticker} met criteria with score {analysis['score']}")
                            results.append({
                                'ticker': ticker,
                                'score': analysis['score']
                            })
                        else:
                            logger.info(f"{ticker} did not meet criteria: {analysis.get('status')}")
                        raw_data.append(analysis)
                    except Exception as e:
                        logger.error(f"Error analyzing {ticker}: {e}")
                        continue
                
                # Sleep between batches (except after the last batch)
                if i + batch_size < len(tickers):
                    logger.info("Sleeping 45 seconds between batches...")
                    time.sleep(45)
            
            logger.info(f"Fundamental analysis complete. {len(results)} stocks met criteria")
            if results:
                logger.info(f"Stocks that passed: {[r['ticker'] for r in results]}")
            return results, raw_data
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return [], []

    def analyze_stock(self, ticker: str) -> Dict:
        """
        Analyzes a single stock and returns a dictionary with analysis results
        """
        data = self.get_fundamental_data(ticker)
        if data is None:
            return {
                'ticker': ticker,
                'status': 'missing_gross_profit',
                'meets_criteria': False
            }
        
        # Check if we have all required metrics
        if 'missing_metrics' in data:
            return {
                'ticker': ticker,
                'status': 'missing_data',
                'missing_metrics': data['missing_metrics'],
                'raw_metrics': data,
                'meets_criteria': False
            }
        
        score = self.score_fundamentals(data)
        
        return {
            'ticker': ticker,
            'status': 'analyzed',
            'score': score,
            'raw_metrics': data,
            'meets_criteria': score > 0
        } 