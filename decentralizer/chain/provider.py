"""Async EVM chain provider using web3.py 7.x."""

from __future__ import annotations

import asyncio

from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware

from decentralizer.config import get_settings
from decentralizer.chain.registry import ChainConfig, get_chain_config


class ChainProvider:
    """Async web3 provider for any EVM chain."""

    def __init__(self, chain_id: int):
        self.chain_config: ChainConfig = get_chain_config(chain_id)
        settings = get_settings()
        rpc_url = settings.get_rpc_url(chain_id)
        self.w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
        if self.chain_config.is_poa:
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    @property
    def chain_id(self) -> int:
        return self.chain_config.chain_id

    async def get_latest_block_number(self) -> int:
        return await self.w3.eth.block_number

    async def get_block(self, block_number: int, full_transactions: bool = True):
        return await self.w3.eth.get_block(block_number, full_transactions=full_transactions)

    async def get_balance(self, address: str) -> float:
        """Get balance in native token."""
        wei = await self.w3.eth.get_balance(self.w3.to_checksum_address(address))
        return float(self.w3.from_wei(wei, "ether"))

    async def get_transaction(self, tx_hash: str):
        return await self.w3.eth.get_transaction(tx_hash)

    async def get_transaction_receipt(self, tx_hash: str):
        return await self.w3.eth.get_transaction_receipt(tx_hash)
