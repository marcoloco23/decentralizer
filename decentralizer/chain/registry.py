"""Chain registry mapping chain_id to configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainConfig:
    chain_id: int
    name: str
    native_token: str
    is_poa: bool = False
    block_time: float = 12.0  # seconds


CHAINS: dict[int, ChainConfig] = {
    1: ChainConfig(chain_id=1, name="ethereum", native_token="ETH"),
    42161: ChainConfig(chain_id=42161, name="arbitrum", native_token="ETH", is_poa=True, block_time=0.25),
    10: ChainConfig(chain_id=10, name="optimism", native_token="ETH", is_poa=True, block_time=2.0),
    8453: ChainConfig(chain_id=8453, name="base", native_token="ETH", is_poa=True, block_time=2.0),
    137: ChainConfig(chain_id=137, name="polygon", native_token="MATIC", is_poa=True, block_time=2.0),
}

CHAIN_NAME_TO_ID: dict[str, int] = {c.name: c.chain_id for c in CHAINS.values()}


def get_chain_config(chain_id: int) -> ChainConfig:
    if chain_id not in CHAINS:
        raise ValueError(f"Unknown chain_id={chain_id}. Supported: {list(CHAINS.keys())}")
    return CHAINS[chain_id]


def resolve_chain(name_or_id: str | int) -> ChainConfig:
    """Resolve a chain name or ID to its config."""
    if isinstance(name_or_id, int):
        return get_chain_config(name_or_id)
    name = str(name_or_id).lower()
    if name.isdigit():
        return get_chain_config(int(name))
    if name not in CHAIN_NAME_TO_ID:
        raise ValueError(f"Unknown chain '{name}'. Supported: {list(CHAIN_NAME_TO_ID.keys())}")
    return get_chain_config(CHAIN_NAME_TO_ID[name])
