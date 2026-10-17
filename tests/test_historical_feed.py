
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.historical_feed import HistoricalMarketFeed
feed = HistoricalMarketFeed()

market = feed.reset()
print("First day:", list(market.values())[0]["date"])
print("Stocks available:", len(market))
print("RELIANCE data:", market["RELIANCE"])

next_market = feed.step()
print("\nNext day:", list(next_market.values())[0]["date"])
print("Stocks available:", len(next_market))

print("\nTotal trading days:", len(feed.dates))
