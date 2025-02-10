import re
from selenium.webdriver import Chrome
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
import time
import math 
import numpy_financial as npf
import numpy as np
from selenium.webdriver.chrome.options import Options
from pandas import DataFrame
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.service import Service

import yahoo_fin.stock_info as si
import yahoo_fin
import yfinance as yf

import datetime
import json
import os

from datetime import datetime

def initialize_session_state():
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'last_run_date' not in st.session_state:
        st.session_state.last_run_date = None

def Stock_Download():
	op = webdriver.ChromeOptions()
	op.add_experimental_option( "prefs",{'profile.managed_default_content_settings.javascript': 2})

	op.add_argument('dom.disable_open_during_load')
	op.add_argument('browser.popups.showPopupBlocker')
	service = Service(ChromeDriverManager().install())

	driver = webdriver.Chrome(service=service,options=op)

	driver.maximize_window()


	
	#driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
	#driver.get("https://finviz.com/screener.ashx?v=111&f=fa_pe_profitable,sh_price_u5&ft=4&o=price")
	driver.get('https://finviz.com/screener.ashx?v=111&f=fa_debteq_u1%2Cfa_eps5years_pos%2Cfa_grossmargin_o30%2Cfa_netmargin_o15%2Cfa_opermargin_pos%2Cfa_pe_profitable%2Cgeo_usa&ft=4&o=ticker')
	#driver.get('https://finviz.com/screener.ashx?v=111&f=fa_debteq_u1,fa_eps5years_o10,fa_grossmargin_o35,fa_netmargin_o15,fa_pe_u30,fa_roe_o10,geo_usa&ft=4&o=-price')
	sayfa_sayisi = int(driver.find_element(By.ID, "pageSelect").text.split('/ ')[1].split('\n')[0])
	tickers = []
	print("Sayfa Sayisi: ", sayfa_sayisi)
	i=0
	counter = 0

	while(counter<sayfa_sayisi):
		#text = "https://finviz.com/screener.ashx?v=111&f=fa_pe_profitable,sh_price_u5&ft=4&o=price"+"&r="+str(i)+"1"
		text = 'https://finviz.com/screener.ashx?v=111&f=fa_debteq_u1%2Cfa_eps5years_pos%2Cfa_grossmargin_o30%2Cfa_netmargin_o15%2Cfa_opermargin_pos%2Cfa_pe_profitable%2Cgeo_usa&ft=4&o=ticker' + "&r=" +str(i)+"1"
		#text = 'https://finviz.com/screener.ashx?v=111&f=fa_debteq_u1,fa_eps5years_o10,fa_grossmargin_o35,fa_netmargin_o15,fa_pe_u30,fa_roe_o10,geo_usa&ft=4&o=-price' + "&r=" +str(i)+"1"
		driver.get(text)

		ticker = driver.find_elements(By.CLASS_NAME, 'tab-link')
		for upper in ticker:
			if upper.text.isupper():
				tickers.append(upper.text)

		#print(tickers)
    
		counter = counter + 1
		i=i+2
	print('Bulunan hisse sayisi: ', len(tickers))

	return tickers
	#webdriver = "/Users/erdemmeral/Desktop/SIMER MAKINA DATA/Sellers/chromedriver"
	#driver_page = Chrome(webdriver)


def Buffett(son_stock):
	progress_bar = st.progress(0)
	buffett_array = []
	sayac = 0
	total_stocks = len(son_stock)
	
	for stock in son_stock:
		sayac += 1
		# Update progress bar
		progress_bar.progress(sayac/total_stocks)
		
		if(len(son_stock) / float(sayac) < 4 and len(son_stock) / float(sayac) > 3.97):
			print('%25')
		if(len(son_stock) / float(sayac) < 2 and len(son_stock) / float(sayac) > 1.97):
			print('%50')
		if(len(son_stock) / float(sayac) < 1 and len(son_stock) / float(sayac) > 0.90):
			print('%75')
		total_points = 0
		ticker = yf.Ticker(stock)
		#print('Stock: ', stock)
		
		
	# Get the company's income statement data
		
		####income statement
		income_statement = ticker.income_stmt
		#print('burdayiz')

		#print(income_statement)
		try:
			gross_profit = income_statement.loc['Gross Profit'][0]
			#print('Gross Profit: ',gross_profit)
			
			sga = income_statement.loc['Selling General And Administration'][0]
			#print('SGA: ',sga)
			
			#r_and_d = income_statement.loc['Research And Development'][0]
			#print('R&D: ', r_and_d)
			#Depreciation
			try:
				depreciation = income_statement.loc['Reconciled Depreciation'][0]
			except (KeyError):
				try:
					depreciation = income_statement.loc['Depreciation And Amortization In Income Statement'][0]
				except (KeyError):
					continue
			#print('Depreciation: ', depreciation)

			try:
				interest_expense = income_statement.loc['Interest Expense'][0]
			except (KeyError):
				try:
					interest_expense = income_statement.loc['Other Income Expense'][0]
				except (KeyError):
					continue
			#print('Interest Expense: ',interest_expense)

			operating_income = income_statement.loc['Operating Income'][0]
			#print('Operating Income: ', operating_income)
		except (KeyError):
			#print('eksik bilgi')
			continue
			#ters mantik - kosulu saglayan atlanir
		#print('burdayiz')
		if(sga > gross_profit * 0.30):
			#print('girdi')
			pass
		else:
		#	print('puan1')
			total_points += 1
		#print('SGA = ', sga)
		#print('gecti ')

		#if(r_and_d > gross_profit * 0.30):
		#	continue
		#print('gecti ')

		if(depreciation > gross_profit * 0.10):
			pass
		else:
			total_points += 1
		#print('gecti ')
		
		if(interest_expense > operating_income * 0.15):
			pass
		#print('gecti ')
		else:
			total_points += 1
			

		####balance sheet
		pd.set_option('display.max_rows', None)
		balance_sheet = ticker.balance_sheet

		#print(balance_sheet)

		#1
		net_income = income_statement.loc['Net Income'][0]
		#print("Net Income: ", net_income)
		try:
			long_term_debt = balance_sheet.loc['Long Term Debt'][0]

			if(math.isnan(long_term_debt) ):
				try:
					long_term_debt = balance_sheet.loc['Total Non Current Liabilities Net Minority Interest'][0]
					#print('girdi')
				except (KeyError, ValueError):
					#print('valueerror yada keyerror girdi (long term lia)')
					continue
		except (KeyError):
			try:
				long_term_debt = balance_sheet.loc['Long Term Debt And Capital Lease Obligation'][0]
			except (KeyError):
				#print('long term eksik')
				continue
		#print('Long Term Debt: ', long_term_debt)
		if(net_income * 4 < long_term_debt):
			pass
		else:
			total_points += 1

		#2 shareholders' equity ratio
		try:
			#total_assets = balance_sheet.loc['Total Assets'][0]
			#print('Total Assets: ',total_assets)
			total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'][0]
			shareholders_equity = balance_sheet.loc["Stockholders Equity"][0]

			if(math.isnan(shareholders_equity) or shareholders_equity <= 0):
				continue
			#print("Shareholders' Equity: " ,shareholders_equity)

			#print("Shareholders equity / total assets = ", shareholders_equity/total_assets)
		except (KeyError):
			#print('Shareholders equity ratio hatasi')
			continue
		
		if ((total_liabilities / shareholders_equity) > 1):
			pass
		else:
			total_points += 1

		#return on shareholders equity
		net_income = income_statement.loc['Net Income'][0]

		if(net_income / shareholders_equity < 0.20):
			#print('Return on shareholders equity error ')
			pass
		else:
			total_points += 1


		#cashflow
		cash_flow = ticker.cashflow
		#print(cash_flow)	
		
		try:
			capital_expenditure = cash_flow.loc['Capital Expenditure Reported'][0]
		except (KeyError):
			try:
				capital_expenditure = cash_flow.loc['Capital Expenditure'][0]
			except (KeyError):
				continue
		if(capital_expenditure / net_income > 0.50):
			#print('Capital expenditure / net income error')
			pass
		#capital_expen
		else:
			total_points += 1

		


		#print(stocks_current_price)

		if (total_points > 6):
			buffett_array.append(stock)
		else:
			continue
			

	return buffett_array


def save_results(stocks_data):
    """Save the analysis results with timestamp"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create the data structure
    data = {
        'date': today,
        'stocks': stocks_data,
        'active_symbols': [stock['symbol'] for stock in stocks_data]
    }
    
    # Load existing data
    history_file = 'stock_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Check if we already have an entry for today
    for i, entry in enumerate(history):
        if entry['date'] == today:
            # Replace today's entry with new data
            history[i] = data
            break
    else:
        # No entry for today, append new data
        history.append(data)
    
    # Sort history by date
    history.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f)

def show_historical_chart():
    """Display historical price charts with status indicators"""
    if not os.path.exists('stock_history.json'):
        st.warning("No historical data available yet.")
        return
        
    with open('stock_history.json', 'r') as f:
        history = json.load(f)
    
    # Sort history by date
    history.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    
    # Get all unique stocks that have ever appeared
    all_stocks = set()
    for entry in history:
        all_stocks.update(entry['active_symbols'])
    
    # Create two columns for the selection area
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_stock = st.selectbox(
            "Select a stock to view historical data", 
            list(sorted(all_stocks)),
            key='historical_stock_selector'
        )
    
    with col2:
        # Show current status of selected stock
        latest_entry = history[-1]
        if selected_stock in latest_entry['active_symbols']:
            st.success("Currently Active")
        else:
            st.warning("Currently Inactive")
    
    with col3:
        # Show number of days tracked
        stock_days = sum(1 for entry in history if selected_stock in entry['active_symbols'])
        st.info(f"Days Tracked: {stock_days}")
    
    if selected_stock:
        chart_container = st.container()
        with chart_container:
            dates = []
            current_prices = []
            pe_prices = []
            pb_prices = []
            buffett_prices = []
            active_status = []
            
            for entry in history:
                is_active = selected_stock in entry['active_symbols']
                active_status.append(is_active)
                
                stock_data = None
                for stock in entry['stocks']:
                    if stock['symbol'] == selected_stock:
                        stock_data = stock
                        break
                
                dates.append(entry['date'])
                if stock_data:
                    current_prices.append(stock_data['current_price'])
                    pe_prices.append(stock_data['pe_price'])
                    pb_prices.append(stock_data['pb_price'])
                    buffett_prices.append(stock_data['buffett_price'])
                else:
                    current_prices.append(None)
                    pe_prices.append(None)
                    pb_prices.append(None)
                    buffett_prices.append(None)
            
            # Create the plot
            fig = go.Figure()
            
            # Add price traces
            fig.add_trace(go.Scatter(
                x=dates, 
                y=current_prices, 
                name='Current Price',
                line=dict(color='blue'), 
                connectgaps=False
            ))
            fig.add_trace(go.Scatter(
                x=dates, 
                y=pe_prices, 
                name='PE Price',
                line=dict(color='red'), 
                connectgaps=False
            ))
            fig.add_trace(go.Scatter(
                x=dates, 
                y=pb_prices, 
                name='PB Price',
                line=dict(color='green'), 
                connectgaps=False
            ))
            fig.add_trace(go.Scatter(
                x=dates, 
                y=buffett_prices, 
                name='Buffett Price',
                line=dict(color='purple'), 
                connectgaps=False
            ))
            
            # Add background colors
            active_ranges = []
            current_start = None
            current_status = None
            
            for i, status in enumerate(active_status):
                if status != current_status:
                    if current_start is not None:
                        active_ranges.append((current_start, dates[i-1], current_status))
                    current_start = dates[i]
                    current_status = status
            
            if current_start is not None:
                active_ranges.append((current_start, dates[-1], current_status))
            
            for start, end, status in active_ranges:
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor='green' if status else 'red',
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )
            
            # Update layout with daily x-axis
            fig.update_layout(
                title=f'Historical Price Analysis for {selected_stock}',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                showlegend=True,
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d',
                    tickmode='auto',
                    nticks=20
                )
            )
            
            st.plotly_chart(fig)
            
            # Add explanation
            st.info("""
            Chart Background:
            - Green: Periods when the stock met all criteria
            - Red: Periods when the stock didn't meet criteria
            - Gaps in lines indicate periods when the stock was not in the analysis
            """)

def price_cal(stocks):
    results_container = st.container()
    stocks_data = []  # To store results for logging
    
    try:
        # Update ChromeDriver initialization with additional options
        op = webdriver.ChromeOptions()
        op.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})
        op.add_argument('dom.disable_open_during_load')
        op.add_argument('browser.popups.showPopupBlocker')
        op.add_argument('--no-sandbox')
        op.add_argument('--disable-dev-shm-usage')
        op.add_argument('--ignore-certificate-errors')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=op)

        driver.maximize_window()
        driver.get('https://finviz.com/groups.ashx?g=sector&v=120&o=pe')
        
        sectors = driver.find_element(By.CLASS_NAME, 'styled-table-new').text.split('\n')[1:12]
        energy = sectors[0].split(' ')
        financial = sectors[1].split(' ')
        utilities = sectors[2].split(' ')
        basic_materials = sectors[3].split(' ')
        consumer_defensive = sectors[4].split(' ')
        consumer_cyclical = sectors[5].split(' ')
        industrials = sectors[6].split(' ')
        healthcare = sectors[7].split(' ')
        communication_service = sectors[8].split(' ')
        real_estate = sectors[9].split(' ')
        technology = sectors[10].split(' ')

        with st.spinner('Calculating stock prices...'):
            for stock in stocks:
                ticker = yf.Ticker(stock)
                #print('Stock: ', stock)
                
                #balance_sheet = ticker.balance_sheet
                #stakeholder_equity = balance_sheet.loc['Stockholders Equity'][0]
                #print(stakeholder_equity)
                sector = ticker.info['sector']
                if (sector == 'Financial Services'):
                    sector = 'Financial'
                print(sector)
                #ebitda = ticker.info['ebitda']
                #print(ebitda)
                price = ticker.info['currentPrice']
                pe = ticker.info['trailingPE']
                pb = price / ticker.info['bookValue']
                #print('Current Price: ',price)
                for i in sectors:
                    for x in i.split(' '):
                        if (x == sector):
                            #print(sectors.index(i))
                            #sector_mc = float((i.split(' ')[2][:-1])) * 1000000000
                            #print(sector_mc)
                            sector_pe = i.split(' ')[3]
                            sector_pb = i.split(' ')[7]
                            

                ###1

                estimated_price_pe = float(price) * float(sector_pe) / float(pe)

                estimated_price_pb = float(price) * float(sector_pb) / float(pb)
                
                income_statement = ticker.income_stmt
                avg_yield = 0.05
                eps_diluted = income_statement.loc['Diluted EPS'][0]

                stocks_worth_price = eps_diluted / avg_yield

                if(price < estimated_price_pe and price < estimated_price_pb and price < stocks_worth_price):
                    stock_data = {
                        'symbol': stock,
                        'current_price': price,
                        'pe_price': estimated_price_pe,
                        'pb_price': estimated_price_pb,
                        'buffett_price': stocks_worth_price
                    }
                    stocks_data.append(stock_data)
                    
                    with results_container:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Stock", stock)
                        col2.metric("Current Price", f"${price:.2f}")
                        col3.metric("PE Price", f"${estimated_price_pe:.2f}")
                        col4.metric("PB Price", f"${estimated_price_pb:.2f}")
                        col5.metric("Buffett Price", f"${stocks_worth_price:.2f}")
        
        # Save results after analysis
        save_results(stocks_data)
        
        # Store results in session state
        st.session_state.analysis_results = stocks_data
        
    except Exception as e:
        st.error(f"Error in price calculation: {str(e)}")
    
    finally:
        try:
            driver.quit()
        except:
            pass

def get_sector_data():
    # Use the same webdriver setup as Stock_Download
    op = webdriver.ChromeOptions()
    op.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})
    op.add_argument('dom.disable_open_during_load')
    op.add_argument('browser.popups.showPopupBlocker')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=op)
    
    # Navigate to Finviz sector page
    driver.get('https://finviz.com/groups.ashx?g=sector&v=110&o=-name')
    
    sectors = []
    try:
        # Find the sector table
        rows = driver.find_elements(By.CSS_SELECTOR, "table.table-light.table-hover.cursor-pointer tr")[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 8:  # Make sure we have enough columns
                sector_name = cols[1].text
                market_cap = cols[2].text
                pe = cols[3].text
                pb = cols[7].text
                
                # Format the data similar to your original structure
                sector_data = f"{sector_name} {market_cap} {pe} {pb}"
                sectors.append(sector_data)
    
    finally:
        driver.quit()
    
    return sectors

# Main app layout
def main():
	st.title("Stock Analysis Tool")
	
	# Initialize session state
	initialize_session_state()
	
	# Create tabs for different views
	tab1, tab2 = st.tabs(["Current Analysis", "Historical Data"])
	
	with tab1:
		st.sidebar.header("Controls")
		if st.sidebar.button("Run Analysis"):
			with st.spinner('Running analysis...'):
				# Run your analysis
				stocks = Stock_Download()
				st.write(f"Found {len(stocks)} stocks matching criteria")
				
				final_stocks = Buffett(stocks)
				st.write(f"Found {len(final_stocks)} stocks matching Buffett criteria")
				
				# Store results in session state
				st.session_state.analysis_run = True
				st.session_state.last_run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				
				# Run price calculation
				price_cal(final_stocks)
		
		# Show last run time if available
		if st.session_state.last_run_date:
			st.sidebar.info(f"Last analysis run: {st.session_state.last_run_date}")
		
		# Display current analysis results if available
		if st.session_state.analysis_run:
			display_current_analysis()
	
	with tab2:
		show_historical_data()

def display_current_analysis():
	"""Display the current analysis results"""
	if os.path.exists('stock_history.json'):
		with open('stock_history.json', 'r') as f:
			history = json.load(f)
			if history:
				latest_results = history[-1]  # Get most recent results
				
				st.subheader("Current Analysis Results")
				for stock in latest_results['stocks']:
					col1, col2, col3, col4, col5 = st.columns(5)
					col1.metric("Stock", stock['symbol'])
					col2.metric("Current Price", f"${stock['current_price']:.2f}")
					col3.metric("PE Price", f"${stock['pe_price']:.2f}")
					col4.metric("PB Price", f"${stock['pb_price']:.2f}")
					col5.metric("Buffett Price", f"${stock['buffett_price']:.2f}")

def show_historical_data():
	"""Display historical data view"""
	if not os.path.exists('stock_history.json'):
		st.warning("No historical data available yet.")
		return
		
	with open('stock_history.json', 'r') as f:
		history = json.load(f)
	
	# Get all unique stocks
	all_stocks = set()
	for entry in history:
		all_stocks.update(entry['active_symbols'])
	
	# Create columns for the selection area
	col1, col2 = st.columns([3, 1])
	
	with col1:
		selected_stock = st.selectbox(
			"Select a stock to view historical data",
			list(sorted(all_stocks)),
			key='historical_stock_selector'
		)
	
	with col2:
		# Show current status
		if selected_stock:
			latest_entry = history[-1]
			if selected_stock in latest_entry['active_symbols']:
				st.success("Currently Active")
			else:
				st.warning("Currently Inactive")
	
	if selected_stock:
		show_stock_history(selected_stock, history)

def show_stock_history(selected_stock, history):
    """Display historical chart for selected stock"""
    chart_container = st.container()
    
    with chart_container:
        # Gather data for the selected stock
        dates = []
        current_prices = []
        pe_prices = []
        pb_prices = []
        buffett_prices = []
        dcf_prices = []
        active_status = []
        
        for entry in history:
            is_active = selected_stock in entry['active_symbols']
            active_status.append(is_active)
            
            stock_data = next(
                (stock for stock in entry['stocks'] if stock['symbol'] == selected_stock),
                None
            )
            
            dates.append(entry['date'])
            if stock_data:
                current_prices.append(stock_data['current_price'])
                pe_prices.append(stock_data['pe_price'])
                pb_prices.append(stock_data['pb_price'])
                buffett_prices.append(stock_data['buffett_price'])
                dcf_prices.append(stock_data.get('dcf_price', None))
            else:
                current_prices.append(None)
                pe_prices.append(None)
                pb_prices.append(None)
                buffett_prices.append(None)
                dcf_prices.append(None)
        
        # Create the plot
        fig = go.Figure()
        
        # Add price traces
        fig.add_trace(go.Scatter(x=dates, y=current_prices, name='Current Price',
                                line=dict(color='blue'), connectgaps=False))
        fig.add_trace(go.Scatter(x=dates, y=pe_prices, name='PE Price',
                                line=dict(color='red'), connectgaps=False))
        fig.add_trace(go.Scatter(x=dates, y=pb_prices, name='PB Price',
                                line=dict(color='green'), connectgaps=False))
        fig.add_trace(go.Scatter(x=dates, y=buffett_prices, name='Buffett Price',
                                line=dict(color='purple'), connectgaps=False))
        fig.add_trace(go.Scatter(x=dates, y=dcf_prices, name='DCF Price',
                                line=dict(color='orange'), connectgaps=False))
        
        # Add background colors to show active/inactive periods
        active_ranges = []
        current_start = None
        current_status = None
        
        for i, status in enumerate(active_status):
            if status != current_status:
                if current_start is not None:
                    active_ranges.append((current_start, dates[i-1], current_status))
                current_start = dates[i]
                current_status = status
        
        # Add the last range
        if current_start is not None:
            active_ranges.append((current_start, dates[-1], current_status))
        
        # Add colored background for active/inactive periods
        for start, end, status in active_ranges:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor='green' if status else 'red',
                opacity=0.1,
                layer="below",
                line_width=0,
            )
        
        fig.update_layout(
            title=f'Historical Price Analysis for {selected_stock}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Add explanation
        st.info("""
        Chart Background:
        - Green: Periods when the stock met all criteria
        - Red: Periods when the stock didn't meet criteria
        - Gaps in lines indicate periods when the stock was not in the analysis
        """)

if __name__ == "__main__":
	main()


