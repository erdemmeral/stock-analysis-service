import os
from dotenv import load_dotenv

load_dotenv()

# Analysis intervals
TECHNICAL_ANALYSIS_INTERVAL = int(os.getenv('ANALYSIS_INTERVAL', '3600'))  # 1 hour in seconds
FUNDAMENTAL_ANALYSIS_INTERVAL = 86400  # 24 hours in seconds

# Scoring thresholds
PORTFOLIO_THRESHOLD_SCORE = int(os.getenv('PORTFOLIO_THRESHOLD_SCORE', '75'))

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')

# Portfolio API URL
PORTFOLIO_API_URL = os.getenv('PORTFOLIO_API_URL', 'https://portfolio-tracker-rough-dawn-5271.fly.dev/api')

# API Keys
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Other configuration... 