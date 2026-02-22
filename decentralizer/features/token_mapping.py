"""Map Numerai crypto symbols to on-chain contract addresses via CoinGecko."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# CoinGecko platform IDs → our chain_id
COINGECKO_PLATFORM_TO_CHAIN: dict[str, int] = {
    "ethereum": 1,
    "arbitrum-one": 42161,
    "optimistic-ethereum": 10,
    "base": 8453,
    "polygon-pos": 137,
}

# Symbols that are native tokens (no ERC-20 address)
NATIVE_TOKENS: dict[str, dict[str, str | int]] = {
    "ETH": {"coingecko_id": "ethereum", "chain_id": 1, "token_address": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},
    "MATIC": {"coingecko_id": "matic-network", "chain_id": 137, "token_address": "0x0000000000000000000000000000000000001010"},
}

# CoinGecko supplementary data fields
SUPPLEMENTARY_FIELDS = [
    "developer_score",
    "community_score",
    "liquidity_score",
    "public_interest_score",
]


async def build_token_mapping(
    conn: duckdb.DuckDBPyConnection,
    live_parquet_path: str | Path,
    rate_limit: float = 3.0,
) -> pd.DataFrame:
    """Map all Numerai symbols to on-chain addresses via CoinGecko.

    1. Read crypto_live.parquet for (symbol, ucid) pairs
    2. Use CoinGecko /coins/list to resolve coingecko_id
    3. Use CoinGecko /coins/{id} to get contract addresses
    4. Store in token_mapping table

    Args:
        conn: DuckDB connection
        live_parquet_path: Path to Numerai crypto_live.parquet
        rate_limit: Seconds between CoinGecko API calls (free tier ~10-30/min)

    Returns:
        DataFrame of mapped tokens
    """
    from decentralizer.storage.database import upsert_token_mapping

    # Read Numerai symbols
    live_df = pd.read_parquet(live_parquet_path)
    if "symbol" not in live_df.columns and live_df.index.name == "symbol":
        live_df = live_df.reset_index()

    symbols = live_df[["symbol"]].drop_duplicates()
    if "ucid" in live_df.columns:
        symbols = live_df[["symbol", "ucid"]].drop_duplicates()
    else:
        symbols["ucid"] = None

    logger.info(f"Found {len(symbols)} Numerai symbols to map")

    # Fetch CoinGecko coin list (no rate limit, single call)
    async with httpx.AsyncClient(timeout=30) as client:
        coins_list = await _fetch_coins_list(client)

    # Build symbol → coingecko_id mapping
    cg_by_symbol = _build_symbol_lookup(coins_list)

    # Resolve each symbol
    mappings = []
    async with httpx.AsyncClient(timeout=30) as client:
        for _, row in symbols.iterrows():
            sym = row["symbol"].upper()
            ucid = row.get("ucid")

            # Check native tokens first
            if sym in NATIVE_TOKENS:
                native = NATIVE_TOKENS[sym]
                mappings.append({
                    "symbol": sym,
                    "ucid": ucid,
                    "coingecko_id": native["coingecko_id"],
                    "chain_id": native["chain_id"],
                    "token_address": native["token_address"],
                    "is_native": True,
                })
                continue

            # Resolve via CoinGecko
            cg_id = cg_by_symbol.get(sym.lower())
            if not cg_id:
                logger.debug(f"No CoinGecko match for {sym}")
                continue

            # Get contract addresses
            await asyncio.sleep(rate_limit)
            addresses = await _fetch_coin_addresses(client, cg_id)
            if not addresses:
                logger.debug(f"No contract addresses for {sym} ({cg_id})")
                continue

            for chain_id, token_address in addresses:
                mappings.append({
                    "symbol": sym,
                    "ucid": ucid,
                    "coingecko_id": cg_id,
                    "chain_id": chain_id,
                    "token_address": token_address.lower(),
                    "is_native": False,
                })

    if not mappings:
        logger.warning("No token mappings resolved")
        return pd.DataFrame()

    mapping_df = pd.DataFrame(mappings)

    # Ensure correct column order for DB
    mapping_df = mapping_df[["symbol", "ucid", "coingecko_id", "chain_id", "token_address", "is_native"]]

    upsert_token_mapping(conn, mapping_df)
    logger.info(f"Mapped {mapping_df['symbol'].nunique()} symbols to {len(mapping_df)} chain addresses")

    return mapping_df


async def fetch_supplementary_data(
    conn: duckdb.DuckDBPyConnection,
    rate_limit: float = 3.0,
) -> pd.DataFrame:
    """Fetch CoinGecko supplementary scores for mapped tokens.

    For tokens without on-chain data (non-EVM), these provide backup features:
    developer_score, community_score, liquidity_score, public_interest_score.

    Returns:
        DataFrame with symbol, coingecko_id, and supplementary score columns
    """
    from decentralizer.storage.database import get_token_mapping

    mapping_df = get_token_mapping(conn)
    if mapping_df.empty:
        return pd.DataFrame()

    # Get unique coingecko_ids
    cg_ids = mapping_df[["symbol", "coingecko_id"]].drop_duplicates(subset=["coingecko_id"])

    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        for _, row in cg_ids.iterrows():
            cg_id = row["coingecko_id"]
            if not cg_id:
                continue

            await asyncio.sleep(rate_limit)
            try:
                resp = await client.get(f"{COINGECKO_BASE}/coins/{cg_id}")
                if resp.status_code != 200:
                    continue
                data = resp.json()
                result = {"symbol": row["symbol"], "coingecko_id": cg_id}
                for field in SUPPLEMENTARY_FIELDS:
                    result[field] = data.get(field, 0) or 0
                results.append(result)
            except Exception as e:
                logger.debug(f"Error fetching supplementary data for {cg_id}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


async def _fetch_coins_list(client: httpx.AsyncClient) -> list[dict]:
    """Fetch the full CoinGecko coins list (free, no key needed)."""
    resp = await client.get(f"{COINGECKO_BASE}/coins/list", params={"include_platform": "true"})
    resp.raise_for_status()
    return resp.json()


def _build_symbol_lookup(coins_list: list[dict]) -> dict[str, str]:
    """Build symbol → coingecko_id mapping, preferring higher-ranked coins.

    CoinGecko lists coins roughly by market cap, so first match wins.
    """
    lookup: dict[str, str] = {}
    for coin in coins_list:
        sym = coin.get("symbol", "").lower()
        if sym and sym not in lookup:
            lookup[sym] = coin["id"]
    return lookup


async def _fetch_coin_addresses(
    client: httpx.AsyncClient,
    coingecko_id: str,
) -> list[tuple[int, str]]:
    """Fetch contract addresses for a coin from CoinGecko.

    Returns list of (chain_id, token_address) tuples for supported chains.
    """
    try:
        resp = await client.get(f"{COINGECKO_BASE}/coins/{coingecko_id}")
        if resp.status_code == 429:
            logger.warning("CoinGecko rate limited, waiting 60s...")
            await asyncio.sleep(60)
            resp = await client.get(f"{COINGECKO_BASE}/coins/{coingecko_id}")
        if resp.status_code != 200:
            return []

        data = resp.json()
        platforms = data.get("platforms", {})

        addresses = []
        for platform, address in platforms.items():
            if not address:
                continue
            chain_id = COINGECKO_PLATFORM_TO_CHAIN.get(platform)
            if chain_id:
                addresses.append((chain_id, address))

        return addresses

    except Exception as e:
        logger.debug(f"Error fetching addresses for {coingecko_id}: {e}")
        return []
