"""Centralized configuration via pydantic-settings. All secrets from .env."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Blockchain RPC
    infura_api_key: str = ""
    etherscan_api_key: str = ""

    # Optional per-chain RPC overrides
    ethereum_rpc_url: str = ""
    arbitrum_rpc_url: str = ""
    optimism_rpc_url: str = ""
    base_rpc_url: str = ""
    polygon_rpc_url: str = ""

    # LLM
    anthropic_api_key: str = ""

    # Data paths
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    duckdb_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "decentralizer.duckdb")

    # Graph algorithm defaults
    damping: float = 0.85
    max_change: float = 0.001
    max_iter: int = 25
    output_limit: int = 100

    # Legacy CSV paths (for migration)
    @property
    def financial_csv(self) -> Path:
        return self.data_dir / "financial_transactions.csv"

    @property
    def non_financial_csv(self) -> Path:
        return self.data_dir / "non_financial_transactions.csv"

    def get_rpc_url(self, chain_id: int) -> str:
        """Get RPC URL for a chain, using override or Infura default."""
        overrides = {
            1: self.ethereum_rpc_url,
            42161: self.arbitrum_rpc_url,
            10: self.optimism_rpc_url,
            8453: self.base_rpc_url,
            137: self.polygon_rpc_url,
        }
        if overrides.get(chain_id):
            return overrides[chain_id]
        # Infura fallback for supported chains
        infura_slugs = {
            1: "mainnet",
            42161: "arbitrum-mainnet",
            10: "optimism-mainnet",
            8453: "base-mainnet",
            137: "polygon-mainnet",
        }
        slug = infura_slugs.get(chain_id)
        if slug and self.infura_api_key:
            return f"https://{slug}.infura.io/v3/{self.infura_api_key}"
        raise ValueError(f"No RPC URL configured for chain_id={chain_id}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
