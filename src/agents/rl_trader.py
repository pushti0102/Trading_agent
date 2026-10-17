
import numpy as np


STATE_SIZE = 10
ORDER_SIZE = 5


def encode_state(observation, initial_cash=100_000, max_order_size=100):
    """Convert one stock's observation into 10 normalized features."""
    price = float(observation["price"])
    fundamental = max(float(observation["fundamental_value"]), 1e-8)
    portfolio_value = max(
        float(observation.get("portfolio_value", initial_cash)), 1e-8
    )

    state = [
        np.clip(price / fundamental - 1.0, -1.0, 1.0),
        np.clip(float(observation["return"]), -0.2, 0.2),
        np.log1p(max(float(observation["volume"]), 0.0)) / 20.0,
        float(observation["spread"]),
        np.log1p(max(float(observation["liquidity"]), 0.0)) / 30.0,
        np.clip(float(observation["volatility"]), 0.0, 0.2),
        float(observation["order_imbalance"]),
        float(observation["inventory"]) / max_order_size,
        float(observation["cash"]) / initial_cash,
        (
            float(observation["inventory"]) * price
        ) / portfolio_value,
    ]

    return np.asarray(state, dtype=np.float32)


def action_to_order(action, inventory, order_size=ORDER_SIZE):
    """Convert DQN action index into a simulated order."""
    if action == 0:
        return "BUY", order_size

    if action == 1:
        return "HOLD", 0

    if action == 2:
        if inventory >= order_size:
            return "SELL", order_size
        return "HOLD", 0

    raise ValueError(f"Invalid action index: {action}")
