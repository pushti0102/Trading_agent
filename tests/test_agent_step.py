
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.market_environment import MarketEnvironment

env = MarketEnvironment(initial_cash=100_000)

result = env.step_agents({
    "leader": {
        "symbol": "RELIANCE",
        "side": "BUY",
        "quantity": 10,
    },
    "rl_trader": {
        "symbol": "INFY",
        "side": "BUY",
        "quantity": 5,
    },
})

print("Executions:")
for agent, execution in result["executions"].items():
    print(agent, execution["status"])

print("\nPortfolio values:", result["portfolio_values"])
print("Next market available:", result["next_market"] is not None)
print("Current date:", next(iter(env.market.values()))["date"])
