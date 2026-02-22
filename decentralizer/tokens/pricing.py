"""Historical token prices via DeFiLlama (free, no API key)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pandas as pd
import duckdb

from decentralizer.tokens.constants import DEFILLAMA_CHAIN_PREFIX, STABLECOINS

DEFILLAMA_BASE = "https://coins.llama.fi"


async def fetch_price_defillama(
    chain_id: int,
    token_address: str,
    timestamp: int,
) -> float | None:
    """Fetch historical price from DeFiLlama for a single token at a timestamp."""
    prefix = DEFILLAMA_CHAIN_PREFIX.get(chain_id)
    if not prefix:
        return None

    coin_id = f"{prefix}:{token_address}"
    url = f"{DEFILLAMA_BASE}/prices/historical/{timestamp}/{coin_id}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
            coins = data.get("coins", {})
            info = coins.get(coin_id)
            if info and "price" in info:
                return float(info["price"])
    except Exception:
        pass
    return None


async def fetch_prices_batch(
    chain_id: int,
    token_addresses: list[str],
    timestamp: int,
) -> dict[str, float]:
    """Fetch prices for multiple tokens at once using DeFiLlama batch endpoint."""
    prefix = DEFILLAMA_CHAIN_PREFIX.get(chain_id)
    if not prefix:
        return {}

    coins = ",".join(f"{prefix}:{addr}" for addr in token_addresses)
    url = f"{DEFILLAMA_BASE}/prices/historical/{timestamp}/{coins}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            data = resp.json()
            results = {}
            for addr in token_addresses:
                coin_id = f"{prefix}:{addr}"
                info = data.get("coins", {}).get(coin_id)
                if info and "price" in info:
                    results[addr] = float(info["price"])
            return results
    except Exception:
        return {}


async def backfill_prices(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    rate_limit: float = 1.0,
) -> int:
    """Backfill historical prices for all tokens seen in token_transfers.

    Strategy:
    1. Get unique (token_address, date) pairs from token_transfers
    2. Skip pairs already in token_prices
    3. Skip known stablecoins (hardcode price = 1.0)
    4. Fetch missing prices from DeFiLlama
    """
    # Insert stablecoin prices first
    stables = STABLECOINS.get(chain_id, {})
    if stables:
        stable_dates = conn.execute("""
            SELECT DISTINCT token_address, CAST(epoch_ms(timestamp * 1000) AS DATE) as date
            FROM token_transfers
            WHERE chain_id = ? AND token_address IN (SELECT UNNEST(?::VARCHAR[]))
              AND timestamp > 0
        """, [chain_id, list(stables.keys())]).fetchdf()

        if not stable_dates.empty:
            stable_dates["chain_id"] = chain_id
            stable_dates["price_usd"] = 1.0
            stable_dates["source"] = "hardcoded"
            stable_prices = stable_dates[["chain_id", "token_address", "date", "price_usd", "source"]]
            conn.execute("INSERT OR IGNORE INTO token_prices SELECT * FROM stable_prices")

    # Get all unique (token, date) pairs that need prices
    missing = conn.execute("""
        WITH needed AS (
            SELECT DISTINCT
                token_address,
                CAST(epoch_ms(timestamp * 1000) AS DATE) as date,
                MIN(timestamp) as sample_ts
            FROM token_transfers
            WHERE chain_id = ? AND timestamp > 0
            GROUP BY token_address, CAST(epoch_ms(timestamp * 1000) AS DATE)
        )
        SELECT n.token_address, n.date, n.sample_ts
        FROM needed n
        LEFT JOIN token_prices tp
          ON tp.chain_id = ? AND tp.token_address = n.token_address AND tp.date = n.date
        WHERE tp.price_usd IS NULL
        ORDER BY n.date
    """, [chain_id, chain_id]).fetchdf()

    if missing.empty:
        return 0

    inserted = 0
    # Group by date to batch API calls
    for date, group in missing.groupby("date"):
        addresses = group["token_address"].tolist()
        sample_ts = int(group["sample_ts"].iloc[0])

        prices = await fetch_prices_batch(chain_id, addresses, sample_ts)

        if prices:
            rows = []
            for addr, price in prices.items():
                rows.append({
                    "chain_id": chain_id,
                    "token_address": addr,
                    "date": date,
                    "price_usd": price,
                    "source": "defillama",
                })
            df = pd.DataFrame(rows)
            conn.execute("INSERT OR IGNORE INTO token_prices SELECT * FROM df")
            inserted += len(rows)

        await asyncio.sleep(rate_limit)

    return inserted
