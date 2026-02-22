"""Hardcoded known contracts — DEX routers, WETH, stablecoins, CEX wallets."""

# address -> (label, category)
KNOWN_LABELS: dict[int, dict[str, tuple[str, str]]] = {
    1: {
        # Stablecoins
        "0xdac17f958d2ee523a2206206994597c13d831ec7": ("Tether USDT", "token"),
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USD Coin USDC", "token"),
        "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", "token"),
        # WETH
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH", "token"),
        # DEX routers
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": ("Uniswap V2 Router", "dex"),
        "0xe592427a0aece92de3edee1f18e0157c05861564": ("Uniswap V3 Router", "dex"),
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": ("Uniswap V3 Router 2", "dex"),
        "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b": ("Uniswap Universal Router", "dex"),
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": ("Uniswap Universal Router V2", "dex"),
        "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": ("SushiSwap Router", "dex"),
        # CEX hot wallets
        "0x28c6c06298d514db089934071355e5743bf21d60": ("Binance Hot Wallet 14", "cex"),
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549": ("Binance Hot Wallet 15", "cex"),
        "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": ("Binance Hot Wallet 16", "cex"),
        "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": ("Binance Hot Wallet 17", "cex"),
        "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": ("Coinbase Commerce", "cex"),
        "0x503828976d22510aad0201ac7ec88293211d23da": ("Coinbase 1", "cex"),
        "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740": ("Coinbase 2", "cex"),
        "0x3cd751e6b0078be393132286c442345e68ff0aaa": ("Coinbase 4", "cex"),
        "0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511": ("Coinbase 5", "cex"),
        "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2": ("FTX Exchange", "cex"),
        "0xc098b2a3aa256d2140208c3de6543aaef5cd3a94": ("FTX Exchange 2", "cex"),
        # Bridges
        "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf": ("Polygon Bridge", "bridge"),
        "0xa3a7b6f88361f48403514059f1f16c8e78d60eec": ("Arbitrum Bridge", "bridge"),
        "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1": ("Optimism Bridge", "bridge"),
        # Uniswap V2 Factory
        "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f": ("Uniswap V2 Factory", "dex"),
        # Uniswap V3 Factory
        "0x1f98431c8ad98523631ae4a59f267346ea31f984": ("Uniswap V3 Factory", "dex"),
    },
    42161: {
        "0x82af49447d8a07e3bd95bd0d56f35241523fbab1": ("WETH", "token"),
        "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9": ("USDT", "token"),
        "0xaf88d065e77c8cc2239327c5edb3a432268e5831": ("USDC", "token"),
        "0xe592427a0aece92de3edee1f18e0157c05861564": ("Uniswap V3 Router", "dex"),
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": ("SushiSwap Router", "dex"),
    },
    10: {
        "0x4200000000000000000000000000000000000006": ("WETH", "token"),
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": ("USDT", "token"),
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85": ("USDC", "token"),
        "0xe592427a0aece92de3edee1f18e0157c05861564": ("Uniswap V3 Router", "dex"),
    },
    8453: {
        "0x4200000000000000000000000000000000000006": ("WETH", "token"),
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": ("USDC", "token"),
        "0x2626664c2603336e57b271c5c0b26f421741e481": ("Uniswap V3 Router 2", "dex"),
    },
    137: {
        "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": ("WMATIC", "token"),
        "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": ("USDT", "token"),
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": ("USDC", "token"),
        "0xe592427a0aece92de3edee1f18e0157c05861564": ("Uniswap V3 Router", "dex"),
        "0xa5e0829caced8ffdd4de3c43696c57f7d7a678ff": ("QuickSwap Router", "dex"),
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": ("SushiSwap Router", "dex"),
    },
}


def get_label(chain_id: int, address: str) -> tuple[str, str] | None:
    """Get (label, category) for a known address, or None."""
    chain_labels = KNOWN_LABELS.get(chain_id, {})
    return chain_labels.get(address.lower())


def is_dex_router(chain_id: int, address: str) -> bool:
    chain_labels = KNOWN_LABELS.get(chain_id, {})
    entry = chain_labels.get(address.lower())
    return entry is not None and entry[1] == "dex"


def is_cex(chain_id: int, address: str) -> bool:
    chain_labels = KNOWN_LABELS.get(chain_id, {})
    entry = chain_labels.get(address.lower())
    return entry is not None and entry[1] == "cex"
