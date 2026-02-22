"""Reconstruct DEX trades from token transfer patterns."""

from __future__ import annotations

import pandas as pd
import duckdb

from decentralizer.tokens.constants import DEX_ROUTERS


def identify_dex_trades(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
) -> pd.DataFrame:
    """Reconstruct DEX trades by analyzing token transfers to/from known DEX routers.

    A DEX trade typically produces 2+ Transfer events in the same tx:
    1. Token A: trader -> router/pair
    2. Token B: router/pair -> trader

    Groups transfers by tx_hash and identifies swaps through known DEX contracts.
    """
    routers = DEX_ROUTERS.get(chain_id, {})
    if not routers:
        return pd.DataFrame()

    router_list = ",".join(f"'{r}'" for r in routers.keys())

    # Find all transfers in txs that touch DEX routers
    trades_df = conn.execute(f"""
        WITH dex_txs AS (
            SELECT DISTINCT tx_hash
            FROM token_transfers
            WHERE chain_id = ?
              AND (from_address IN ({router_list}) OR to_address IN ({router_list}))
        ),
        tx_transfers AS (
            SELECT t.*
            FROM token_transfers t
            JOIN dex_txs d ON t.tx_hash = d.tx_hash
            WHERE t.chain_id = ?
        )
        SELECT * FROM tx_transfers ORDER BY tx_hash, log_index
    """, [chain_id, chain_id]).fetchdf()

    if trades_df.empty:
        return pd.DataFrame()

    router_set = set(routers.keys())
    trade_rows: list[dict] = []

    for tx_hash, group in trades_df.groupby("tx_hash"):
        trade_rows.extend(_reconstruct_trade(tx_hash, group, routers, router_set, chain_id))

    return pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()


def _reconstruct_trade(
    tx_hash: str,
    transfers: pd.DataFrame,
    routers: dict[str, str],
    router_set: set[str],
    chain_id: int,
) -> list[dict]:
    """Given all transfers in a single tx, identify the swap trade."""
    all_addrs = set(transfers["from_address"]) | set(transfers["to_address"])
    trader_candidates = all_addrs - router_set

    # Zero address is minting, not a trader
    trader_candidates.discard("0x" + "0" * 40)

    trades: list[dict] = []
    for trader in trader_candidates:
        sent = transfers[transfers["from_address"] == trader]
        received = transfers[transfers["to_address"] == trader]

        if sent.empty or received.empty:
            continue

        # Use the first sent and first received token as the trade pair
        token_in_row = sent.iloc[0]
        token_out_row = received.iloc[0]

        # Skip if same token (not a swap)
        if token_in_row["token_address"] == token_out_row["token_address"]:
            continue

        # Determine which DEX
        dex = "unknown"
        for addr in router_set & all_addrs:
            if addr in routers:
                dex = routers[addr]
                break

        amount_in = float(token_in_row.get("value_decimal") or 0)
        amount_out = float(token_out_row.get("value_decimal") or 0)

        trades.append({
            "chain_id": chain_id,
            "tx_hash": tx_hash,
            "log_index": int(token_in_row["log_index"]),
            "block_number": int(token_in_row["block_number"]),
            "timestamp": int(token_in_row["timestamp"]),
            "dex": dex,
            "trader": trader,
            "token_in": token_in_row["token_address"],
            "token_out": token_out_row["token_address"],
            "amount_in": amount_in,
            "amount_out": amount_out,
            "amount_usd": None,  # Filled by price backfill
        })

    return trades


def backfill_trade_usd(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> int:
    """Backfill amount_usd on dex_trades using token_prices."""
    conn.execute("""
        UPDATE dex_trades
        SET amount_usd = COALESCE(
            dt.amount_in * tp_in.price_usd,
            dt.amount_out * tp_out.price_usd
        )
        FROM dex_trades dt
        LEFT JOIN token_prices tp_in
          ON tp_in.chain_id = dt.chain_id
          AND tp_in.token_address = dt.token_in
          AND tp_in.date = CAST(epoch_ms(dt.timestamp * 1000) AS DATE)
        LEFT JOIN token_prices tp_out
          ON tp_out.chain_id = dt.chain_id
          AND tp_out.token_address = dt.token_out
          AND tp_out.date = CAST(epoch_ms(dt.timestamp * 1000) AS DATE)
        WHERE dex_trades.chain_id = ?
          AND dex_trades.amount_usd IS NULL
          AND dex_trades.tx_hash = dt.tx_hash
          AND dex_trades.log_index = dt.log_index
    """, [chain_id])
    return 0
