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
                'required_max': 2.0,  # Tightened from 2.5
                'max': 1.5,          # Tightened from 1.8
                'weight': 1.3        # Increased weight
            },
            'eps_growth_5y': {
                'required_min': -10,  # Tightened from -15
                'min': 0,            # Tightened from -5
                'weight': 1.3        # Increased weight
            },
            'gross_margin': {
                'required_min': 20,   # Increased from 15
                'min': 30,           # Increased from 25
                'weight': 1.2
            },
            'net_margin': {
                'required_min': 0,    # Tightened from -2
                'min': 5,            # Increased from 2
                'weight': 1.3        # Increased weight
            },
            'operating_margin': {
                'required_min': 0,    # Tightened from -5
                'min': 5,            # Increased from 0
                'weight': 1.2
            }
        }
        
        # Buffett criteria - tightened but still modern
        self.buffett_metrics = {
            'sga_to_gross_profit': {
                'required_max': 0.50,  # Tightened from 0.60
                'max': 0.40,          # Tightened from 0.45
                'weight': 1.0
            },
            'depreciation_to_gross_profit': {
                'required_max': 0.20,  # Tightened from 0.25
                'max': 0.15,          # Tightened from 0.20
                'weight': 0.9
            },
            'interest_to_operating_income': {
                'required_max': 0.30,  # Tightened from 0.35
                'max': 0.20,          # Tightened from 0.25
                'weight': 1.1
            },
            'debt_coverage': {
                'required_min': 2.5,   # Increased from 2.0
                'min': 3.5,           # Increased from 3.0
                'weight': 1.2
            },
            'leverage_ratio': {
                'required_max': 1.8,   # Tightened from 2.0
                'max': 1.3,           # Tightened from 1.5
                'weight': 1.1
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
            
            # Basic metrics with validation
            basic_metrics = {}
            
            # Check each critical metric with more lenient handling
            critical_metrics = {
                'debt_to_equity': ('debtToEquity', None),
                'eps_growth_5y': ('earningsGrowth', None),
                'gross_margin': ('grossMargins', None),
                'net_margin': ('profitMargins', None),
                'operating_margin': ('operatingMargins', None),
                'pe_ratio': ('trailingPE', None)
            }
            
            for metric, (info_key, default) in critical_metrics.items():
                value = info.get(info_key, default)
                if value is None or value in [float('inf'), float('-inf')]:
                    basic_metrics[metric] = None
                else:
                    value = value * 100 if metric == 'eps_growth_5y' else value
                    basic_metrics[metric] = value
            
            # Get raw financial data
            raw_data = self._get_raw_financial_data(income_stmt, balance_sheet)
            basic_metrics.update(raw_data)
            
            # Calculate additional metrics if possible, with error handling
            try:
                if raw_data.get('gross_profit') and raw_data.get('sga'):
                    basic_metrics['sga_to_gross_profit'] = min(1.0, raw_data['sga'] / raw_data['gross_profit'])
            except:
                pass

            try:
                if raw_data.get('gross_profit') and raw_data.get('depreciation'):
                    basic_metrics['depreciation_to_gross_profit'] = min(1.0, raw_data['depreciation'] / raw_data['gross_profit'])
            except:
                pass

            try:
                if raw_data.get('operating_income') and raw_data.get('interest_expense'):
                    basic_metrics['interest_to_operating_income'] = min(1.0, raw_data['interest_expense'] / abs(raw_data['operating_income']))
            except:
                pass

            try:
                if raw_data.get('net_income') and raw_data.get('long_term_debt'):
                    basic_metrics['debt_coverage'] = (raw_data['net_income'] * 4) / raw_data['long_term_debt']
            except:
                pass

            try:
                if raw_data.get('total_liabilities') and raw_data.get('shareholders_equity'):
                    basic_metrics['leverage_ratio'] = raw_data['total_liabilities'] / abs(raw_data['shareholders_equity'])
            except:
                pass
            
            return basic_metrics
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
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
        """Calculate fundamental score with balanced criteria"""
        try:
            # Check minimum criteria
            if not self.meets_minimum_criteria(data):
                return 0
            
            # Require minimum data quality
            if sum(1 for metric, value in data.items() if value is not None) < 4:
                return 0
            
            scores = []
            weights = []
            
            # Score each fundamental metric
            for metric, criteria in self.fundamental_metrics.items():
                value = data.get(metric)
                if value is None:
                    continue
                
                min_val = criteria.get('required_min', criteria.get('min', float('-inf')))
                max_val = criteria.get('required_max', criteria.get('max', float('inf')))
                target_min = criteria.get('min', min_val)
                target_max = criteria.get('max', max_val)
                weight = criteria['weight']
                
                # Calculate score - stricter scoring
                if value < min_val or value > max_val:
                    score = 25  # Increased penalty
                elif target_min <= value <= target_max:
                    score = 100
                else:
                    # Scale score based on how close to ideal range
                    if value < target_min:
                        score = 40 + (60 * (value - min_val) / (target_min - min_val))
                    else:
                        score = 40 + (60 * (max_val - value) / (max_val - target_max))
                
                scores.append(score * weight)
                weights.append(weight)
            
            # Score Buffett criteria
            buffett_scores = []
            buffett_weights = []
            
            for metric, criteria in self.buffett_metrics.items():
                value = data.get(metric)
                if value is None:
                    continue
                
                min_val = criteria.get('required_min', criteria.get('min', float('-inf')))
                max_val = criteria.get('required_max', criteria.get('max', float('inf')))
                target_min = criteria.get('min', min_val)
                target_max = criteria.get('max', max_val)
                weight = criteria['weight']
                
                if value < min_val or value > max_val:
                    score = 25  # Increased penalty
                elif target_min <= value <= target_max:
                    score = 100
                else:
                    if value < target_min:
                        score = 40 + (60 * (value - min_val) / (target_min - min_val))
                    else:
                        score = 40 + (60 * (max_val - value) / (max_val - target_max))
                
                buffett_scores.append(score * weight)
                buffett_weights.append(weight)
            
            # Calculate final score
            basic_score = sum(scores) / sum(weights) if weights else 0
            buffett_score = sum(buffett_scores) / sum(buffett_weights) if buffett_weights else 50
            
            # Combined score: 80% basic metrics, 20% Buffett criteria (adjusted weights)
            final_score = (basic_score * 0.80) + (buffett_score * 0.20)
            
            # Stricter passing threshold
            return final_score if final_score >= 65 else 0  # Increased from 60
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 0

    def meets_minimum_criteria(self, data: Dict) -> bool:
        """
        Checks if stock meets minimum required criteria with balanced flexibility
        """
        try:
            # Require minimum number of metrics to be present
            available_metrics = sum(1 for metric, value in data.items() if value is not None)
            if available_metrics < 4:  # At least 4 basic metrics must be present
                return False
            
            # Count how many metrics pass minimum criteria
            passed_metrics = 0
            total_metrics = 0
            
            # Check basic screening criteria
            for metric, criteria in self.fundamental_metrics.items():
                value = data.get(metric)
                if value is None:
                    continue
                
                total_metrics += 1
                
                if 'required_min' in criteria and value < criteria['required_min']:
                    continue
                if 'required_max' in criteria and value > criteria['required_max']:
                    continue
                    
                passed_metrics += 1
            
            # Check Buffett criteria
            buffett_passed = 0
            buffett_total = 0
            
            for metric, criteria in self.buffett_metrics.items():
                value = data.get(metric)
                if value is None:
                    continue
                
                buffett_total += 1
                
                if 'required_min' in criteria and value < criteria['required_min']:
                    continue
                if 'required_max' in criteria and value > criteria['required_max']:
                    continue
                    
                buffett_passed += 1
            
            # Require at least 55% of available metrics to pass
            basic_ratio = passed_metrics / total_metrics if total_metrics > 0 else 0
            buffett_ratio = buffett_passed / buffett_total if buffett_total > 0 else 0
            
            # Weight the ratios (75% basic, 25% Buffett)
            final_ratio = (basic_ratio * 0.75) + (buffett_ratio * 0.25)
            
            return final_ratio >= 0.55  # Increased from 0.50
            
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
                            stock_result = {
                                'ticker': ticker,
                                'score': analysis['score']
                            }
                            results.append(stock_result)
                            raw_data.append(analysis)
                            
                            # Trigger immediate analysis for this stock
                            try:
                                # Use asyncio.create_task to run in background
                                import asyncio
                                from src.service.analysis_service import AnalysisService
                                
                                async def analyze_passing_stock():
                                    service = AnalysisService()
                                    await service.analyze_single_stock(ticker)
                                
                                # Run in background without waiting
                                asyncio.create_task(analyze_passing_stock())
                                logger.info(f"Triggered immediate analysis for {ticker}")
                            except Exception as e:
                                logger.error(f"Error triggering immediate analysis for {ticker}: {e}")
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