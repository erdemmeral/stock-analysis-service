import logging
from typing import Dict, List
import asyncpg
from datetime import datetime
from .config import DATABASE_URL

logger = logging.getLogger(__name__)

class Database:
    async def init(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(DATABASE_URL)
        await self._create_tables()
    
    async def _create_tables(self):
        """Create necessary database tables"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    ticker TEXT PRIMARY KEY,
                    fundamental_score FLOAT,
                    last_updated TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS portfolio (
                    ticker TEXT PRIMARY KEY,
                    entry_price FLOAT,
                    entry_date TIMESTAMP,
                    combined_score FLOAT,
                    analysis JSONB
                );
                
                CREATE TABLE IF NOT EXISTS analysis_history (
                    ticker TEXT,
                    timestamp TIMESTAMP,
                    fundamental_score FLOAT,
                    technical_score FLOAT,
                    news_score FLOAT,
                    combined_score FLOAT,
                    analysis_data JSONB
                );
            """)

    async def update_watchlist(self, stocks: List[Dict]):
        """Update watchlist with fundamental analysis results"""
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO watchlist (ticker, fundamental_score, last_updated)
                VALUES ($1, $2, $3)
                ON CONFLICT (ticker) 
                DO UPDATE SET 
                    fundamental_score = EXCLUDED.fundamental_score,
                    last_updated = EXCLUDED.last_updated
            """, [(s['ticker'], s['score'], datetime.now()) for s in stocks])

    async def get_watchlist(self) -> List[str]:
        """Get current watchlist tickers"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT ticker FROM watchlist")
            return [row['ticker'] for row in rows]

    async def is_in_portfolio(self, ticker: str) -> bool:
        """Check if stock is in portfolio"""
        async with self.pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM portfolio WHERE ticker = $1)",
                ticker
            )
            return exists

    async def add_to_portfolio(self, ticker: str, data: Dict):
        """Add stock to portfolio"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO portfolio (ticker, entry_price, entry_date, combined_score, analysis)
                VALUES ($1, $2, $3, $4, $5)
            """, ticker, data['entry_price'], data['entry_date'], 
                data['score'], data['analysis'])

    async def remove_from_portfolio(self, ticker: str):
        """Remove stock from portfolio"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM portfolio WHERE ticker = $1",
                ticker
            )

    async def store_analysis_result(self, ticker: str, timestamp: datetime, analysis_data: Dict):
        """Store analysis results"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO analysis_history 
                (ticker, timestamp, fundamental_score, technical_score, 
                 news_score, combined_score, analysis_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, ticker, timestamp, analysis_data['fundamental_score'],
                analysis_data['technical_score'], analysis_data['news_score'],
                analysis_data['combined_score'], analysis_data) 