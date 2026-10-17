
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.execution_engine import ExecutionEngine

engine = ExecutionEngine(max_order_size=100)

buy = engine.execute(
    agent_id="leader",
    symbol="RELIANCE",
    side="BUY",
    quantity=10,
    reference_price=1000,
    available_cash=100_000,
    available_inventory=0,
)
print("BUY:", buy)

sell = engine.execute(
    agent_id="leader",
    symbol="RELIANCE",
    side="SELL",
    quantity=5,
    reference_price=1000,
    available_cash=100_000,
    available_inventory=10,
)
print("\nSELL:", sell)

large_order = engine.execute(
    agent_id="leader",
    symbol="RELIANCE",
    side="BUY",
    quantity=101,
    reference_price=1000,
    available_cash=1_000_000,
    available_inventory=0,
)
print("\nOversized order:", large_order)

print("\nSuccessful execution records:", len(engine.records))
