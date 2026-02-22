"""Parallel block fetcher with rate limiting."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pandas as pd
from tqdm import tqdm

from decentralizer.chain.provider import ChainProvider
from decentralizer.models.schema import Transaction, Block


class BlockFetcher:
    """Fetch blocks and transactions in parallel with semaphore rate limiting."""

    def __init__(self, chain_id: int, max_concurrent: int = 10):
        self.provider = ChainProvider(chain_id)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.chain_id = chain_id

    async def fetch_block(self, block_number: int) -> tuple[Block, list[Transaction]]:
        """Fetch a single block and its transactions."""
        async with self.semaphore:
            block = await self.provider.get_block(block_number, full_transactions=True)

        block_model = Block(
            chain_id=self.chain_id,
            number=block["number"],
            timestamp=block["timestamp"],
            transaction_count=len(block["transactions"]),
        )

        transactions = []
        for tx in block["transactions"]:
            sender = tx.get("from", "")
            receiver = tx.get("to", "")
            if not sender or not receiver:
                continue

            # Detect EIP-1559
            max_fee = tx.get("maxFeePerGas")
            max_priority = tx.get("maxPriorityFeePerGas")
            tx_type = tx.get("type", 0)
            if isinstance(tx_type, str):
                tx_type = int(tx_type, 16)

            value = float(self.provider.w3.from_wei(tx["value"], "ether"))

            # Convert HexBytes fields to strings
            tx_hash = tx["hash"]
            tx_hash = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)
            input_data = tx.get("input", "0x")
            if hasattr(input_data, "hex"):
                input_data = input_data.hex()
            input_data = str(input_data)[:256]

            transactions.append(Transaction(
                chain_id=self.chain_id,
                hash=tx_hash,
                block_number=block["number"],
                sender=sender.lower(),
                receiver=receiver.lower(),
                value=value,
                timestamp=block["timestamp"],
                gas=tx["gas"],
                gas_price=tx.get("gasPrice") or tx.get("effectiveGasPrice") or 0,
                max_fee_per_gas=max_fee,
                max_priority_fee_per_gas=max_priority,
                input_data=input_data,
                tx_type=tx_type,
            ))

        return block_model, transactions

    async def fetch_blocks(
        self,
        start_block: int,
        end_block: int,
        progress: bool = True,
    ) -> tuple[list[Block], list[Transaction]]:
        """Fetch a range of blocks in parallel."""
        block_numbers = list(range(start_block, end_block + 1))
        all_blocks: list[Block] = []
        all_transactions: list[Transaction] = []

        tasks = [self.fetch_block(n) for n in block_numbers]

        desc = f"Fetching blocks {start_block}-{end_block}"
        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=desc,
            disable=not progress,
        ):
            result = await coro
            results.append(result)

        for block, txs in results:
            all_blocks.append(block)
            all_transactions.extend(txs)

        return all_blocks, all_transactions

    async def fetch_latest(
        self, num_blocks: int = 100, progress: bool = True
    ) -> tuple[list[Block], list[Transaction]]:
        """Fetch the latest N blocks."""
        latest = await self.provider.get_latest_block_number()
        start = latest - num_blocks + 1
        return await self.fetch_blocks(start, latest, progress=progress)

    def blocks_to_dataframe(self, blocks: list[Block]) -> pd.DataFrame:
        return pd.DataFrame([b.model_dump() for b in blocks])

    def transactions_to_dataframe(self, transactions: list[Transaction]) -> pd.DataFrame:
        return pd.DataFrame([t.model_dump() for t in transactions])
