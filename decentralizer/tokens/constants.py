"""Event signatures, known DEX routers, WETH/stablecoin addresses."""

# ERC-20 Transfer(address indexed from, address indexed to, uint256 value)
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Uniswap V2 Swap(address,uint256,uint256,uint256,uint256,address)
UNISWAP_V2_SWAP_TOPIC = (
    "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
)

# Uniswap V3 Swap(address,address,int256,int256,uint160,uint128,int24)
UNISWAP_V3_SWAP_TOPIC = (
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
)

# Minimal ERC-20 ABI for on-chain metadata calls
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]

# WETH (or wrapped native token) per chain
WETH_ADDRESSES: dict[int, str] = {
    1: "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    42161: "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
    10: "0x4200000000000000000000000000000000000006",
    8453: "0x4200000000000000000000000000000000000006",
    137: "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",  # WMATIC
}

# Top stablecoins: address -> (symbol, decimals)
STABLECOINS: dict[int, dict[str, tuple[str, int]]] = {
    1: {
        "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT", 6),
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC", 6),
        "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", 18),
        "0x4fabb145d64652a948d72533023f6e7a623c7c53": ("BUSD", 18),
        "0x8e870d67f660d95d5be530380d0ec0bd388289e1": ("USDP", 18),
    },
    42161: {
        "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9": ("USDT", 6),
        "0xaf88d065e77c8cc2239327c5edb3a432268e5831": ("USDC", 6),
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": ("DAI", 18),
    },
    10: {
        "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": ("USDT", 6),
        "0x0b2c639c533813f4aa9d7837caf62653d097ff85": ("USDC", 6),
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": ("DAI", 18),
    },
    8453: {
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": ("USDC", 6),
        "0x50c5725949a6f0c72e6c4a641f24049a917db0cb": ("DAI", 18),
    },
    137: {
        "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": ("USDT", 6),
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": ("USDC", 6),
        "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063": ("DAI", 18),
    },
}

# DEX router contracts: address -> dex name
DEX_ROUTERS: dict[int, dict[str, str]] = {
    1: {
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "uniswap_v2",
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3",
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3_router2",
        "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b": "uniswap_universal",
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "uniswap_universal_v2",
        "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "sushiswap",
    },
    42161: {
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": "sushiswap",
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3",
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3_router2",
    },
    10: {
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3",
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3_router2",
    },
    8453: {
        "0x2626664c2603336e57b271c5c0b26f421741e481": "uniswap_v3_router2",
    },
    137: {
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3",
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": "sushiswap",
        "0xa5e0829caced8ffdd4de3c43696c57f7d7a678ff": "quickswap",
    },
}

# DeFiLlama chain prefixes for price API
DEFILLAMA_CHAIN_PREFIX: dict[int, str] = {
    1: "ethereum",
    42161: "arbitrum",
    10: "optimism",
    8453: "base",
    137: "polygon",
}

# CoinGecko platform names → chain_id
COINGECKO_CHAIN_PREFIX: dict[int, str] = {
    1: "ethereum",
    42161: "arbitrum-one",
    10: "optimistic-ethereum",
    8453: "base",
    137: "polygon-pos",
}
