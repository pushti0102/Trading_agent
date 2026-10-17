
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.market_environment import MarketEnvironment

env = MarketEnvironment()

# Inspect one stock's observation for the leader.
state = env.observe("leader", "RELIANCE")

required = [
    "price",
    "return",
    "volume",
    "spread",
    "liquidity",
    "volatility",
    "order_imbalance",
    "inventory",
    "cash",
    "fundamental_value",
]

print("=== RELIANCE AGENT STATE ===")
for key in required:
    print(f"{key}: {state[key]}")

assert all(key in state for key in required)
assert len(env.observe("leader")["stocks"]) == 10

print("\nPASS: All 10 state variables are present")
print("PASS: All 10 stocks are available")
