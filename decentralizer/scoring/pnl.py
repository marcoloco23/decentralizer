"""Wallet P&L calculation using FIFO cost basis."""

from __future__ import annotations

import time
from collections import defaultdict, deque

import pandas as pd
import duckdb


def calculate_wallet_pnl(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
) -> pd.DataFrame:
    """Calculate P&L for a single wallet using FIFO cost basis.

    1. Get all token transfers for this wallet, ordered by timestamp
    2. For each token, maintain a FIFO queue of (quantity, cost_per_unit)
    3. On buy (transfer in): push to queue with price at that time
    4. On sell (transfer out): pop from queue, compute realized P&L
    5. Unrealized P&L = current_holdings * latest_price - remaining_cost_basis
    """
    transfers = conn.execute("""
        SELECT
            tt.token_address,
            tt.from_address,
            tt.to_address,
            tt.value_decimal,
            tt.timestamp,
            tt.log_index,
            COALESCE(tp.price_usd, 0) as price_usd
        FROM token_transfers tt
        LEFT JOIN token_prices tp
          ON tt.chain_id = tp.chain_id
          AND tt.token_address = tp.token_address
          AND tp.date = CAST(epoch_ms(tt.timestamp * 1000) AS DATE)
        WHERE tt.chain_id = ?
          AND (tt.from_address = ? OR tt.to_address = ?)
          AND tt.value_decimal IS NOT NULL
          AND tt.value_decimal > 0
        ORDER BY tt.timestamp, tt.log_index
    """, [chain_id, address, address]).fetchdf()

    if transfers.empty:
        return pd.DataFrame()

    # FIFO calculation per token
    holdings: dict[str, deque] = defaultdict(deque)
    realized: dict[str, float] = defaultdict(float)

    for _, row in transfers.iterrows():
        token = row["token_address"]
        qty = float(row["value_decimal"])
        price = float(row["price_usd"])

        if row["to_address"] == address:
            # Buy / receive
            if qty > 0:
                holdings[token].append((qty, price))
        elif row["from_address"] == address:
            # Sell / send
            remaining = qty
            while remaining > 0 and holdings[token]:
                lot_qty, lot_cost = holdings[token][0]
                if lot_qty <= remaining:
                    holdings[token].popleft()
                    realized[token] += lot_qty * (price - lot_cost)
                    remaining -= lot_qty
                else:
                    holdings[token][0] = (lot_qty - remaining, lot_cost)
                    realized[token] += remaining * (price - lot_cost)
                    remaining = 0

    # Build results
    all_tokens = set(list(realized.keys()) + list(holdings.keys()))
    now = int(time.time())
    results: list[dict] = []

    for token in all_tokens:
        current_qty = sum(q for q, _ in holdings.get(token, []))
        cost_basis = sum(q * c for q, c in holdings.get(token, []))

        latest_price = _get_latest_price(conn, chain_id, token)
        unrealized = current_qty * (latest_price or 0) - cost_basis

        results.append({
            "chain_id": chain_id,
            "address": address,
            "token_address": token,
            "cost_basis": cost_basis,
            "quantity": current_qty,
            "realized_pnl": realized.get(token, 0.0),
            "unrealized_pnl": unrealized,
            "total_pnl": realized.get(token, 0.0) + unrealized,
            "last_updated": now,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def _get_latest_price(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
) -> float | None:
    result = conn.execute("""
        SELECT price_usd FROM token_prices
        WHERE chain_id = ? AND token_address = ?
        ORDER BY date DESC LIMIT 1
    """, [chain_id, token_address]).fetchone()
    return result[0] if result else None


def calculate_all_wallet_pnl(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    min_transfers: int = 5,
    progress: bool = True,
) -> int:
    """Batch compute P&L for all wallets with >= min_transfers token transfers."""
    from tqdm import tqdm

    wallets = conn.execute("""
        SELECT address, COUNT(*) as cnt FROM (
            SELECT from_address as address FROM token_transfers WHERE chain_id = ?
            UNION ALL
            SELECT to_address as address FROM token_transfers WHERE chain_id = ?
        )
        GROUP BY address
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, [chain_id, chain_id, min_transfers]).fetchdf()

    if wallets.empty:
        return 0

    total = 0
    for addr in tqdm(wallets["address"], desc="Computing wallet P&L", disable=not progress):
        pnl_df = calculate_wallet_pnl(conn, addr, chain_id)
        if not pnl_df.empty:
            conn.execute("INSERT OR REPLACE INTO wallet_pnl SELECT * FROM pnl_df")
            total += len(pnl_df)

    return total
