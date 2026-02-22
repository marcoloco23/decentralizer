"""Pydantic v2 data models with multi-chain support."""

from pydantic import BaseModel, Field


class Block(BaseModel):
    chain_id: int = Field(default=1, description="EVM chain ID")
    number: int
    timestamp: int
    transaction_count: int


class Transaction(BaseModel):
    chain_id: int = Field(default=1, description="EVM chain ID")
    hash: str
    block_number: int
    sender: str
    receiver: str
    value: float = Field(description="Value in native token (ETH, MATIC, etc.)")
    timestamp: int
    gas: int
    gas_price: int
    # EIP-1559 fields
    max_fee_per_gas: int | None = None
    max_priority_fee_per_gas: int | None = None
    input_data: str = ""
    tx_type: int = Field(default=0, description="0=legacy, 2=EIP-1559")


class Address(BaseModel):
    chain_id: int = Field(default=1)
    address: str
    page_rank: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    total_received: float = 0.0
    total_sent: float = 0.0
    tx_count: int = 0


class AddressMetrics(BaseModel):
    chain_id: int = Field(default=1)
    address: str
    page_rank: float = 0.0
    weighted_page_rank: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    influence_score: float = 0.0
    community_id: int = -1
    anomaly_score: float = 0.0
