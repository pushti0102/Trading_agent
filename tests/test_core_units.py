import numpy as np

from src.environment.execution_engine import ExecutionEngine
from src.agents.rl_trader import STATE_SIZE, action_to_order, encode_state


def test_buy_execution_includes_spread_impact_and_fee():
    engine = ExecutionEngine(
        transaction_cost=0.001,
        max_order_size=100,
        impact_coefficient=0.0005,
        spread_rate=0.02,
    )
    result = engine.execute(
        agent_id="tester",
        symbol="ABC",
        side="BUY",
        quantity=10,
        reference_price=100.0,
        available_cash=10_000.0,
        available_inventory=0,
    )
    assert result["status"] == "EXECUTED"
    assert result["execution_price"] > 100.0
    assert result["fee"] > 0
    assert result["quantity"] == 10


def test_sell_rejects_insufficient_inventory():
    engine = ExecutionEngine()
    result = engine.execute(
        agent_id="tester",
        symbol="ABC",
        side="SELL",
        quantity=2,
        reference_price=100.0,
        available_cash=1_000.0,
        available_inventory=1,
    )
    assert result["status"] == "REJECTED"
    assert result["reason"] == "Insufficient inventory"


def test_oversized_order_is_rejected():
    engine = ExecutionEngine(max_order_size=10)
    result = engine.execute(
        agent_id="tester",
        symbol="ABC",
        side="BUY",
        quantity=11,
        reference_price=100.0,
        available_cash=10_000.0,
        available_inventory=0,
    )
    assert result["status"] == "REJECTED"


def test_state_encoder_returns_ten_finite_features():
    observation = {
        "price": 101.0,
        "fundamental_value": 100.0,
        "portfolio_value": 100_000.0,
        "return": 0.01,
        "volume": 1000,
        "spread": 0.02,
        "liquidity": 1_000_000.0,
        "volatility": 0.02,
        "order_imbalance": 0.0,
        "inventory": 5,
        "cash": 99_000.0,
    }
    state = encode_state(observation)
    assert state.shape == (STATE_SIZE,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()


def test_invalid_sell_action_maps_to_hold():
    assert action_to_order(2, inventory=0) == ("HOLD", 0)
