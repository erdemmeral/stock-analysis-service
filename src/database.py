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
                    technical_score FLOAT,
                    technical_scores JSONB DEFAULT '{}',
                    news_score FLOAT,
                    news_sentiment TEXT,
                    risk_level TEXT,
                    current_price FLOAT,
                    last_updated TIMESTAMP,
                    last_news_check TIMESTAMP,
                    last_technical_check TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS portfolio (
                    ticker TEXT PRIMARY KEY,
                    entry_price FLOAT,
                    current_price FLOAT,
                    entry_date TIMESTAMP,
                    combined_score FLOAT,
                    fundamental_score FLOAT,
                    technical_score FLOAT,
                    news_score FLOAT,
                    analysis JSONB
                );
                
                CREATE TABLE IF NOT EXISTS analysis_history (
                    ticker TEXT,
                    timestamp TIMESTAMP,
                    fundamental_score FLOAT,
                    technical_score FLOAT,
                    news_score FLOAT,
                    combined_score FLOAT,
                    analysis_data JSONB,
                    PRIMARY KEY (ticker, timestamp)
                );
            """)

    async def update_watchlist(self, stocks: List[Dict]):
        """Update watchlist with fundamental analysis results"""
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO watchlist (
                    ticker, 
                    fundamental_score, 
                    technical_score,
                    news_score,
                    news_sentiment,
                    risk_level,
                    last_updated
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ticker) 
                DO UPDATE SET 
                    fundamental_score = EXCLUDED.fundamental_score,
                    technical_score = EXCLUDED.technical_score,
                    news_score = EXCLUDED.news_score,
                    news_sentiment = EXCLUDED.news_sentiment,
                    risk_level = EXCLUDED.risk_level,
                    last_updated = EXCLUDED.last_updated
            """, [(
                s['ticker'], 
                s['score'],
                s.get('technical_score', 50),
                s.get('news_score', 50),
                s.get('news_sentiment', 'neutral'),
                s.get('risk_level', 'medium'),
                datetime.now()
            ) for s in stocks])

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

    async def update_watchlist_item(self, ticker: str, update_data: Dict):
        """Update a single watchlist item"""
        try:
            async with self.pool.acquire() as conn:
                # First check if item exists
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM watchlist WHERE ticker = $1)",
                    ticker
                )
                
                if not exists:
                    # Insert new item if it doesn't exist
                    await conn.execute("""
                        INSERT INTO watchlist (
                            ticker,
                            technical_score,
                            technical_scores,
                            news_score,
                            news_sentiment,
                            risk_level,
                            current_price,
                            last_updated,
                            last_news_check,
                            last_technical_check
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, 
                        ticker,
                        float(update_data.get('technical_score', 50.0)),
                        update_data.get('technical_scores', {}),
                        float(update_data.get('news_score', 50.0)),
                        update_data.get('news_sentiment', 'neutral'),
                        update_data.get('risk_level', 'medium'),
                        float(update_data.get('current_price', 0.0)),
                        datetime.now(),
                        datetime.now(),
                        datetime.now()
                    )
                else:
                    # Prepare update data with proper type conversion
                    tech_score = update_data.get('technical_score')
                    news_score = update_data.get('news_score')
                    current_price = update_data.get('current_price')
                    
                    # Convert scores to float if they exist
                    if tech_score is not None:
                        tech_score = float(tech_score)
                    if news_score is not None:
                        news_score = float(news_score)
                    if current_price is not None:
                        current_price = float(current_price)
                    
                    # Update existing item
                    await conn.execute("""
                        UPDATE watchlist 
                        SET technical_score = COALESCE($2, technical_score),
                            technical_scores = COALESCE($3, technical_scores),
                            news_score = COALESCE($4, news_score),
                            news_sentiment = COALESCE($5, news_sentiment),
                            risk_level = COALESCE($6, risk_level),
                            current_price = COALESCE($7, current_price),
                            last_updated = $8,
                            last_news_check = CASE 
                                WHEN $4 IS NOT NULL THEN $8 
                                ELSE last_news_check 
                            END,
                            last_technical_check = CASE 
                                WHEN $2 IS NOT NULL THEN $8 
                                ELSE last_technical_check 
                            END
                        WHERE ticker = $1
                    """, 
                        ticker,
                        tech_score,
                        update_data.get('technical_scores', None),
                        news_score,
                        update_data.get('news_sentiment'),
                        update_data.get('risk_level'),
                        current_price,
                        datetime.now()
                    )
                logger.info(f"Successfully updated watchlist item for {ticker} with scores: tech={tech_score}, news={news_score}")
        except Exception as e:
            logger.error(f"Error updating watchlist item for {ticker}: {e}")
            raise 