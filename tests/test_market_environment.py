
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.market_environment import MarketEnvironment
env = MarketEnvironment(initial_cash=100_000)

observation = env.observe("leader")
print("Initial cash:", observation["cash"])
print("Initial portfolio value:", observation["portfolio_value"])

buy = env.submit_order("leader", "RELIANCE", "BUY", 10)
print("\nBuy result:", buy)

observation = env.observe("leader")
print("Cash after buy:", round(observation["cash"], 2))
print("RELIANCE holdings:", observation["inventory"]["RELIANCE"])
print("Portfolio value:", round(observation["portfolio_value"], 2))

sell = env.submit_order("leader", "RELIANCE", "SELL", 5)
print("\nSell result:", sell)

print("\nInsufficient inventory test:")
print(env.submit_order("leader", "RELIANCE", "SELL", 1000))

print("\nAdvance one day:")
env.step()
print("New date:", next(iter(env.market.values()))["date"])
