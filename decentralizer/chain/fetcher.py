"""Parallel block fetcher with rate limiting. Optimized for throughput."""

from __future__ import annotations

import asyncio

import pandas as pd
from tqdm import tqdm

from decentralizer.chain.provider import ChainProvider
from decentralizer.tokens.constants import TRANSFER_TOPIC


class BlockFetcher:
    """Fetch blocks and transactions in parallel with high concurrency."""

    def __init__(self, chain_id: int, max_concurrent: int = 50):
        self.provider = ChainProvider(chain_id)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.chain_id = chain_id

    async def _fetch_block_raw(self, block_number: int) -> dict | None:
        """Fetch a single block. Returns raw dict or None on error."""
        async with self.semaphore:
            try:
                return await self.provider.get_block(block_number, full_transactions=True)
            except Exception:
                return None

    def _parse_block(self, block: dict) -> tuple[dict, list[dict]]:
        """Parse a raw block into a block row and transaction rows (plain dicts, no pydantic)."""
        block_row = {
            "chain_id": self.chain_id,
            "number": block["number"],
            "timestamp": block["timestamp"],
            "transaction_count": len(block["transactions"]),
        }

        tx_rows = []
        ts = block["timestamp"]
        bn = block["number"]
        w3 = self.provider.w3

        for tx in block["transactions"]:
            sender = tx.get("from")
            receiver = tx.get("to")
            if not sender or not receiver:
                continue

            tx_hash = tx["hash"]
            tx_hash = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)

            input_data = tx.get("input", b"0x")
            if hasattr(input_data, "hex"):
                input_data = input_data.hex()
            input_data = str(input_data)[:256]

            tx_type = tx.get("type", 0)
            if isinstance(tx_type, str):
                tx_type = int(tx_type, 16)

            tx_rows.append({
                "chain_id": self.chain_id,
                "hash": tx_hash,
                "block_number": bn,
                "sender": sender.lower(),
                "receiver": receiver.lower(),
                "value": float(w3.from_wei(tx["value"], "ether")),
                "timestamp": ts,
                "gas": tx["gas"],
                "gas_price": tx.get("gasPrice") or tx.get("effectiveGasPrice") or 0,
                "max_fee_per_gas": tx.get("maxFeePerGas"),
                "max_priority_fee_per_gas": tx.get("maxPriorityFeePerGas"),
                "input_data": input_data,
                "tx_type": tx_type,
            })

        return block_row, tx_rows

    async def fetch_blocks(
        self,
        start_block: int,
        end_block: int,
        progress: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch a range of blocks in parallel. Returns (blocks_df, transactions_df)."""
        block_numbers = list(range(start_block, end_block + 1))

        # Fire all requests
        tasks = [self._fetch_block_raw(n) for n in block_numbers]

        all_block_rows: list[dict] = []
        all_tx_rows: list[dict] = []

        desc = f"Fetching blocks {start_block}-{end_block}"
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc, disable=not progress):
            raw = await coro
            if raw is None:
                continue
            block_row, tx_rows = self._parse_block(raw)
            all_block_rows.append(block_row)
            all_tx_rows.extend(tx_rows)

        blocks_df = pd.DataFrame(all_block_rows) if all_block_rows else pd.DataFrame()
        txs_df = pd.DataFrame(all_tx_rows) if all_tx_rows else pd.DataFrame()

        return blocks_df, txs_df

    async def fetch_latest(
        self, num_blocks: int = 100, progress: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch the latest N blocks."""
        latest = await self.provider.get_latest_block_number()
        start = latest - num_blocks + 1
        return await self.fetch_blocks(start, latest, progress=progress)

    # --- ERC-20 token transfer fetching ---

    async def _fetch_logs_batch(self, from_block: int, to_block: int) -> list[dict]:
        """Fetch Transfer event logs for a block range. Splits on error."""
        async with self.semaphore:
            try:
                return await self.provider.get_logs(
                    from_block=from_block,
                    to_block=to_block,
                    topics=[TRANSFER_TOPIC],
                )
            except Exception:
                if to_block - from_block > 0:
                    mid = (from_block + to_block) // 2
                    left = await self._fetch_logs_batch(from_block, mid)
                    right = await self._fetch_logs_batch(mid + 1, to_block)
                    return left + right
                return []

    async def fetch_token_transfers(
        self,
        start_block: int,
        end_block: int,
        batch_size: int = 500,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Fetch ERC-20 Transfer events using eth_getLogs in batches."""
        from decentralizer.tokens.transfers import parse_transfer_logs

        batches = []
        for b in range(start_block, end_block + 1, batch_size):
            batches.append((b, min(b + batch_size - 1, end_block)))

        tasks = [self._fetch_logs_batch(f, t) for f, t in batches]
        all_logs: list[dict] = []

        desc = f"Fetching transfer logs {start_block}-{end_block}"
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=desc,
            disable=not progress,
        ):
            logs = await coro
            all_logs.extend(logs)

        return parse_transfer_logs(all_logs, self.chain_id)
