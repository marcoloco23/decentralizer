"""Etherscan API for contract labels + token metadata. On-chain fallback."""

from __future__ import annotations

import asyncio
import time

import httpx
import pandas as pd
from web3 import AsyncWeb3

from decentralizer.config import get_settings
from decentralizer.tokens.constants import ERC20_ABI

ETHERSCAN_API_URLS: dict[int, str] = {
    1: "https://api.etherscan.io/api",
    42161: "https://api.arbiscan.io/api",
    10: "https://api-optimistic.etherscan.io/api",
    8453: "https://api.basescan.org/api",
    137: "https://api.polygonscan.com/api",
}


async def fetch_token_metadata_onchain(
    w3: AsyncWeb3,
    token_address: str,
) -> dict | None:
    """Fetch ERC-20 symbol/name/decimals via on-chain calls."""
    try:
        checksum = w3.to_checksum_address(token_address)
        contract = w3.eth.contract(address=checksum, abi=ERC20_ABI)
        decimals = await contract.functions.decimals().call()
        try:
            symbol = await contract.functions.symbol().call()
        except Exception:
            symbol = ""
        try:
            name = await contract.functions.name().call()
        except Exception:
            name = ""
        return {
            "address": token_address.lower(),
            "symbol": symbol,
            "name": name,
            "decimals": decimals,
        }
    except Exception:
        return None


async def fetch_token_metadata_etherscan(
    chain_id: int,
    token_address: str,
    api_key: str | None = None,
) -> dict | None:
    """Fetch token info from Etherscan API."""
    base_url = ETHERSCAN_API_URLS.get(chain_id)
    if not base_url:
        return None

    if not api_key:
        settings = get_settings()
        api_key = getattr(settings, "etherscan_api_key", None) or ""

    params = {
        "module": "token",
        "action": "tokeninfo",
        "contractaddress": token_address,
        "apikey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(base_url, params=params)
            data = resp.json()
            if data.get("status") == "1" and data.get("result"):
                info = data["result"][0] if isinstance(data["result"], list) else data["result"]
                return {
                    "address": token_address.lower(),
                    "symbol": info.get("symbol", ""),
                    "name": info.get("tokenName", "") or info.get("name", ""),
                    "decimals": int(info.get("divisor", 18) or 18),
                }
    except Exception:
        pass
    return None


async def resolve_token_metadata(
    chain_id: int,
    token_addresses: list[str],
    w3: AsyncWeb3 | None = None,
    rate_limit: float = 0.25,
) -> pd.DataFrame:
    """Resolve metadata for a list of token addresses. On-chain first, Etherscan fallback.

    Returns DataFrame with columns: chain_id, address, symbol, name, decimals.
    """
    rows: list[dict] = []
    for addr in token_addresses:
        result = None
        if w3:
            result = await fetch_token_metadata_onchain(w3, addr)
        if not result:
            result = await fetch_token_metadata_etherscan(chain_id, addr)
            await asyncio.sleep(rate_limit)
        if result:
            result["chain_id"] = chain_id
            rows.append(result)
        else:
            rows.append({
                "chain_id": chain_id,
                "address": addr.lower(),
                "symbol": "",
                "name": "",
                "decimals": 18,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
