import os
from dotenv import load_dotenv

load_dotenv()

# Analysis intervals
TECHNICAL_ANALYSIS_INTERVAL = int(os.getenv('ANALYSIS_INTERVAL', '3600'))  # 1 hour in seconds
FUNDAMENTAL_ANALYSIS_INTERVAL = 86400  # 24 hours in seconds

# Scoring thresholds
PORTFOLIO_THRESHOLD_SCORE = int(os.getenv('PORTFOLIO_THRESHOLD_SCORE', '75'))

# Telegram configuration
TELEGRAM_BOT_TOKEN = '7530096691:AAEbEnXA2PJyihv0S3HD_5q1_pq6QRSj7-g'
TELEGRAM_CHANNEL_ID = '-4678926083'

# Portfolio API URL
PORTFOLIO_API_URL = 'https://portfolio-tracker-rough-dawn-5271.fly.dev/api'

# API Keys
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key"
FINNHUB_KEY = "your_finnhub_key"
NEWS_API_KEY = "your_newsapi_key"

# Other configuration... 