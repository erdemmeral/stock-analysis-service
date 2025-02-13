import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Analysis intervals
TECHNICAL_ANALYSIS_INTERVAL = int(os.getenv('ANALYSIS_INTERVAL', '900'))  # 15 minutes in seconds
FUNDAMENTAL_ANALYSIS_INTERVAL = 86400  # 24 hours in seconds
NEWS_ANALYSIS_INTERVAL = 300  # 5 minutes in seconds

# Scoring thresholds
PORTFOLIO_THRESHOLD_SCORE = int(os.getenv('PORTFOLIO_THRESHOLD_SCORE', '75'))

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')

# Portfolio API URL
PORTFOLIO_API_URL = os.getenv('PORTFOLIO_API_URL', 'https://portfolio-tracker-rough-dawn-5271.fly.dev/api')

# Analysis timeframes
TIMEFRAME_CONFIGS = {
    'short': {
        'ma_short': 5,      # 1 week
        'ma_long': 20,      # 1 month
        'period': '2mo',    # Changed from '1mo' to '2mo' to ensure enough data points
        'interval': '1d',   # Daily data
        'weight': 0.3       # 30% weight for combined score
    },
    'medium': {
        'ma_short': 20,     # 1 month
        'ma_long': 50,      # ~2.5 months
        'period': '6mo',    # 6 months of data
        'interval': '1d',
        'weight': 0.5       # 50% weight for combined score
    },
    'long': {
        'ma_short': 50,     # ~2.5 months
        'ma_long': 100,     # ~5 months
        'period': '1y',     # 1 year of data
        'interval': '1d',
        'weight': 0.2       # 20% weight for combined score
    }
}

# Add profit target configurations
PROFIT_TARGETS = {
    'low_risk': {
        'first_target': 0.15,    # 15% for first partial exit
        'final_target': 0.25     # 25% for full exit
    },
    'medium_risk': {
        'first_target': 0.20,    # 20% for first partial exit
        'final_target': 0.35     # 35% for full exit
    },
    'high_risk': {
        'first_target': 0.25,    # 25% for first partial exit
        'final_target': 0.50     # 50% for full exit
    }
}

# Trailing stop settings
TRAILING_STOP = {
    'activation': 0.10,          # Start trailing at 10% profit
    'distance': 0.05            # 5% trailing distance
}

# Other configuration... 

def verify_config():
    """Verify all required configurations are present"""
    required_vars = [
        'ANALYSIS_INTERVAL',
        'PORTFOLIO_THRESHOLD_SCORE',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHANNEL_ID',
        'PORTFOLIO_API_URL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    return True

# Add to existing config.py
if not verify_config():
    logger.warning("Some configurations are missing. Check your .env file.") 