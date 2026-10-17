
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.market_environment import MarketEnvironment

env = MarketEnvironment(initial_cash=100_000, max_order_size=100)

symbol = "RELIANCE"
initial_cash = env.cash["leader"]
reference_price = env.last_prices[symbol]

# Buy 10 shares
buy = env.submit_order("leader", symbol, "BUY", 10)
assert buy["status"] == "EXECUTED", buy

expected_cash = initial_cash - buy["trade_value"] - buy["fee"]
assert abs(env.cash["leader"] - expected_cash) < 0.01
assert env.inventory["leader"][symbol] == 10

# Sell 5 shares
sell = env.submit_order("leader", symbol, "SELL", 5)
assert sell["status"] == "EXECUTED", sell

assert env.inventory["leader"][symbol] == 5

expected_cash += sell["trade_value"] - sell["fee"]
assert abs(env.cash["leader"] - expected_cash) < 0.01

# Try selling more shares than held, without exceeding max order size
rejected = env.submit_order("leader", symbol, "SELL", 6)
assert rejected["status"] == "REJECTED"
assert rejected["reason"] == "Insufficient inventory"

# Other agents must remain unaffected
assert env.cash["rl_trader"] == 100_000
assert env.inventory["rl_trader"][symbol] == 0

print("PASS: Buy accounting")
print("PASS: Sell accounting")
print("PASS: Insufficient inventory rejection")
print("PASS: Agent portfolio isolation")
print(f"Remaining cash: ₹{env.cash['leader']:.2f}")
print(f"Remaining {symbol} shares: {env.inventory['leader'][symbol]}")
print(f"Recorded executions: {len(env.history)}")
