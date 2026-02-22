"""Parse ERC-20 Transfer event logs into structured data."""

from __future__ import annotations

import pandas as pd


def parse_transfer_logs(logs: list[dict], chain_id: int) -> pd.DataFrame:
    """Parse raw eth_getLogs results for Transfer events into a DataFrame.

    Transfer event: Transfer(address indexed from, address indexed to, uint256 value)
    - topics[0] = event signature (0xddf252ad...)
    - topics[1] = from address (padded to 32 bytes)
    - topics[2] = to address (padded to 32 bytes)
    - data = value (uint256)
    """
    rows: list[dict] = []
    for log in logs:
        topics = log.get("topics", [])
        if len(topics) < 3:
            continue  # Not a standard ERC-20 Transfer

        tx_hash = log["transactionHash"]
        if hasattr(tx_hash, "hex"):
            tx_hash = "0x" + tx_hash.hex()
        else:
            tx_hash = str(tx_hash)

        token_address = log["address"]
        if hasattr(token_address, "lower"):
            token_address = token_address.lower()

        from_addr = _topic_to_address(topics[1])
        to_addr = _topic_to_address(topics[2])

        data = log.get("data", "0x0")
        if hasattr(data, "hex"):
            data = "0x" + data.hex()
        value_raw = str(int(data, 16)) if data and data not in ("0x", "0x0", "") else "0"

        log_index = log.get("logIndex", 0)
        if isinstance(log_index, str):
            log_index = int(log_index, 16)

        block_number = log["blockNumber"]
        if isinstance(block_number, str):
            block_number = int(block_number, 16)

        rows.append({
            "chain_id": chain_id,
            "tx_hash": tx_hash,
            "log_index": log_index,
            "block_number": block_number,
            "token_address": token_address,
            "from_address": from_addr,
            "to_address": to_addr,
            "value_raw": value_raw,
            "value_decimal": None,
            "timestamp": 0,  # Backfilled from blocks table
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _topic_to_address(topic) -> str:
    """Convert a 32-byte padded topic to a 20-byte hex address."""
    if hasattr(topic, "hex"):
        hex_str = topic.hex()
    else:
        hex_str = str(topic)
        if hex_str.startswith("0x"):
            hex_str = hex_str[2:]
    return "0x" + hex_str[-40:].lower()


def backfill_timestamps(conn, chain_id: int = 1) -> int:
    """Fill in timestamp values from the blocks table for token_transfers."""
    result = conn.execute("""
        UPDATE token_transfers
        SET timestamp = b.timestamp
        FROM blocks b
        WHERE token_transfers.chain_id = b.chain_id
          AND token_transfers.block_number = b.number
          AND token_transfers.chain_id = ?
          AND token_transfers.timestamp = 0
    """, [chain_id])
    return result.fetchone()[0] if result else 0


def backfill_decimals(conn, chain_id: int = 1) -> int:
    """Fill in value_decimal using token_metadata decimals."""
    conn.execute("""
        UPDATE token_transfers
        SET value_decimal = CAST(value_raw AS DOUBLE) / POWER(10, tm.decimals)
        FROM token_metadata tm
        WHERE token_transfers.chain_id = tm.chain_id
          AND token_transfers.token_address = tm.address
          AND token_transfers.chain_id = ?
          AND token_transfers.value_decimal IS NULL
    """, [chain_id])
    return 0
