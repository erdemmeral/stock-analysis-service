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
        # Basic screening metrics with minimum and ideal thresholds
        self.fundamental_metrics = {
            'debt_to_equity': {
                'required_max': 1.5,  # Maximum allowed
                'max': 1.0,          # Ideal maximum
                'weight': 1.0
            },
            'eps_growth_5y': {
                'required_min': -10,  # Minimum allowed
                'min': 0,            # Ideal minimum
                'weight': 1.0
            },
            'gross_margin': {
                'required_min': 20,   # Minimum allowed
                'min': 30,           # Ideal minimum
                'weight': 1.0
            },
            'net_margin': {
                'required_min': 5,    # Minimum allowed
                'min': 15,           # Ideal minimum
                'weight': 1.0
            },
            'operating_margin': {
                'required_min': -5,   # Minimum allowed
                'min': 0,            # Ideal minimum
                'weight': 1.0
            },
            'country': {
                'value': 'USA',
                'weight': 1.0
            }
        }
        
        # Buffett criteria with required and ideal thresholds
        self.buffett_metrics = {
            'sga_to_gross_profit': {
                'required_max': 0.40,  # Maximum allowed
                'max': 0.30,          # Ideal maximum
                'weight': 1.0
            },
            'depreciation_to_gross_profit': {
                'required_max': 0.15,  # Maximum allowed
                'max': 0.10,          # Ideal maximum
                'weight': 1.0
            },
            'interest_to_operating_income': {
                'required_max': 0.20,  # Maximum allowed
                'max': 0.15,          # Ideal maximum
                'weight': 1.0
            },
            'debt_coverage': {
                'required_min': 3.0,   # Minimum allowed
                'min': 4.0,           # Ideal minimum
                'weight': 1.0
            },
            'leverage_ratio': {
                'required_max': 1.2,   # Maximum allowed
                'max': 1.0,           # Ideal maximum
                'weight': 1.0
            }
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
    
    def score_fundamentals(self, data: Dict) -> float:
        """Calculate fundamental score with more stringent criteria"""
        try:
            # First check if stock meets minimum criteria
            if not self.meets_minimum_criteria(data):
                return 0
            
            scores = []
            weights = []
            
            # Score each fundamental metric
            for metric, criteria in self.fundamental_metrics.items():
                if metric == 'country':  # Skip non-numeric criteria
                    continue
                    
                value = data.get(metric)
                if value is None:
                    return 0  # Fail if any required metric is missing
                
                # Get thresholds
                min_val = criteria.get('required_min', criteria.get('min', float('-inf')))
                max_val = criteria.get('required_max', criteria.get('max', float('inf')))
                target_min = criteria.get('min', min_val)
                target_max = criteria.get('max', max_val)
                weight = criteria['weight']
                
                # Calculate score - more stringent scoring
                if value < min_val or value > max_val:
                    return 0  # Fail immediately if any metric is outside required range
                elif target_min <= value <= target_max:
                    score = 100  # Meets ideal criteria
                else:
                    # Scale score based on how close to ideal range - more stringent scaling
                    if value < target_min:
                        score = 60 + (40 * (value - min_val) / (target_min - min_val))
                    else:  # value > target_max
                        score = 60 + (40 * (max_val - value) / (max_val - target_max))
                
                scores.append(score * weight)
                weights.append(weight)
            
            # Score Buffett criteria
            for metric, criteria in self.buffett_metrics.items():
                value = data.get(metric)
                if value is None:
                    return 0  # Fail if any Buffett metric is missing
                    
                min_val = criteria.get('required_min', criteria.get('min', float('-inf')))
                max_val = criteria.get('required_max', criteria.get('max', float('inf')))
                target_min = criteria.get('min', min_val)
                target_max = criteria.get('max', max_val)
                weight = criteria['weight']
                
                # Calculate score - more stringent scoring
                if value < min_val or value > max_val:
                    return 0  # Fail immediately if any metric is outside required range
                elif target_min <= value <= target_max:
                    score = 100
                else:
                    # Scale score based on how close to ideal range - more stringent scaling
                    if value < target_min:
                        score = 60 + (40 * (value - min_val) / (target_min - min_val))
                    else:
                        score = 60 + (40 * (max_val - value) / (max_val - target_max))
                
                scores.append(score * weight)
                weights.append(weight)
            
            # Calculate weighted average score
            if not scores:
                return 0
            
            final_score = sum(scores) / sum(weights)
            
            # Additional threshold - require at least 70% score to pass
            return final_score if final_score >= 70 else 0
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 0

    def meets_minimum_criteria(self, data: Dict) -> bool:
        """
        Checks if stock meets minimum required criteria - more stringent version
        """
        try:
            # Check if we have all required metrics
            required_metrics = set(self.fundamental_metrics.keys()) | set(self.buffett_metrics.keys())
            missing_metrics = [metric for metric in required_metrics if metric not in data]
            if missing_metrics:
                logger.info(f"Missing required metrics: {missing_metrics}")
                return False
            
            # Check basic screening criteria
            for metric, criteria in self.fundamental_metrics.items():
                if metric == 'country':
                    if data[metric] != criteria['value']:
                        logger.info(f"Failed country check: {data[metric]} != {criteria['value']}")
                        return False
                    continue
                    
                value = data[metric]
                if 'required_min' in criteria and value < criteria['required_min']:
                    logger.info(f"Failed {metric} minimum: {value} < {criteria['required_min']}")
                    return False
                if 'required_max' in criteria and value > criteria['required_max']:
                    logger.info(f"Failed {metric} maximum: {value} > {criteria['required_max']}")
                    return False
            
            # Check ALL Buffett criteria
            for metric, criteria in self.buffett_metrics.items():
                value = data[metric]
                if 'required_min' in criteria and value < criteria['required_min']:
                    logger.info(f"Failed Buffett {metric} minimum: {value} < {criteria['required_min']}")
                    return False
                if 'required_max' in criteria and value > criteria['required_max']:
                    logger.info(f"Failed Buffett {metric} maximum: {value} > {criteria['required_max']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking criteria: {e}")
            return False

    def meets_all_criteria(self, data: Dict) -> bool:
        """Check if stock meets minimum required criteria"""
        try:
            # Check fundamental metrics
            for metric, criteria in self.fundamental_metrics.items():
                if metric == 'country':
                    if data.get(metric) != criteria['value']:
                        logger.info(f"Failed {metric}: {data[metric]} != {criteria['value']}")
                        return False
                    continue
                    
                value = data.get(metric)
                if value is None:
                    continue
                    
                min_val = criteria.get('required_min', float('-inf'))
                max_val = criteria.get('required_max', float('inf'))
                
                if value < min_val or value > max_val:
                    logger.info(f"Failed {metric}: {value} not in range [{min_val}, {max_val}]")
                    return False
            
            # Check Buffett criteria
            for metric, criteria in self.buffett_metrics.items():
                value = data.get(metric)
                if value is None:
                    continue
                    
                min_val = criteria.get('required_min', float('-inf'))
                max_val = criteria.get('required_max', float('inf'))
                
                if value < min_val or value > max_val:
                    logger.info(f"Failed Buffett metric {metric}: {value} not in range [{min_val}, {max_val}]")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking criteria: {e}")
            return False

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
            'meets_criteria': score >= 70  # Require at least 70% score to pass
        } 