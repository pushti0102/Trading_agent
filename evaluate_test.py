
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DQNAgent
from src.agents.rl_trader import STATE_SIZE, action_to_order, encode_state
from src.environment.market_environment import MarketEnvironment
from src.game_theory.stackelberg_game import StackelbergGame


INITIAL_CASH = 100_000.0
MODEL_PATH = Path("models/dqn_trader.pt")
HISTORY_PATH = Path("test_portfolio_history.csv")
SUMMARY_PATH = Path("test_evaluation_summary.csv")

CASH_RESERVE_FRACTION = 0.05
MAX_POSITION_FRACTION = 0.20


def restrict_dates(env, dates):
    dates = list(dates)
    env.feed.data = env.feed.data[
        env.feed.data["TIMESTAMP"].isin(dates)
    ].copy()
    env.feed.dates = dates
    env.feed.current_step = 0
    env.reset()


def portfolio_value(env, agent_id):
    return float(
        env.cash[agent_id]
        + sum(
            quantity * env.last_prices[symbol]
            for symbol, quantity in env.inventory[agent_id].items()
        )
    )


def affordable_quantity(env, agent_id, symbol, side, requested, price):
    requested = min(
        int(requested),
        int(env.execution_engine.max_order_size),
    )
    if requested <= 0 or price <= 0:
        return 0

    if side == "SELL":
        return min(
            requested,
            int(env.inventory[agent_id].get(symbol, 0)),
        )

    if side != "BUY":
        return 0

    value = portfolio_value(env, agent_id)
    held = int(env.inventory[agent_id].get(symbol, 0))
    current_price = float(env.last_prices.get(symbol, price))
    position_room = max(
        0.0,
        value * MAX_POSITION_FRACTION - held * current_price,
    )
    max_position_qty = int(position_room / price)

    spendable_cash = max(
        0.0,
        float(env.cash[agent_id]) - INITIAL_CASH * CASH_RESERVE_FRACTION,
    )

    engine = env.execution_engine

    def estimated_cost(quantity):
        adjustment = (
            float(engine.spread_rate) / 2.0
            + float(engine.impact_coefficient) * quantity
        )
        execution_price = price * (1.0 + adjustment)
        trade_value = execution_price * quantity
        return trade_value * (1.0 + float(engine.transaction_cost))

    quantity = min(requested, max_position_qty)
    while quantity > 0 and estimated_cost(quantity) > spendable_cash:
        quantity -= 1

    return quantity


def record_values(env, date, rows):
    for agent_id in ("rl_trader", "leader", "momentum", "buy_hold"):
        observation = env.observe(agent_id)
        rows.append({
            "date": date,
            "agent": agent_id,
            "cash": float(observation["cash"]),
            "portfolio_value": float(observation["portfolio_value"]),
        })


def execute_actions(env, agent_id, actions, next_market):
    for symbol, (side, requested) in actions.items():
        if side == "HOLD" or requested <= 0:
            continue

        price = float(next_market[symbol]["open"])
        quantity = affordable_quantity(
            env, agent_id, symbol, side, requested, price
        )
        if quantity <= 0:
            continue

        env.submit_order(
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=price,
        )


def execute_buy_and_hold(env, first_market):
    """Buy an equal-budget basket on the first test date and then hold."""
    symbols = list(first_market)
    if not symbols:
        return

    target_per_stock = INITIAL_CASH * 0.95 / len(symbols)

    for symbol in symbols:
        price = float(first_market[symbol]["open"])
        quantity = min(
            int(env.execution_engine.max_order_size),
            int(target_per_stock / max(price, 1e-8)),
        )

        while quantity > 0:
            result = env.submit_order(
                agent_id="buy_hold",
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                reference_price=price,
            )
            if result["status"] == "EXECUTED":
                break
            if result.get("reason") != "Insufficient cash":
                break
            quantity -= 1


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}. Run python train_rl.py first."
        )

    reference_env = MarketEnvironment(initial_cash=INITIAL_CASH)
    all_dates = list(reference_env.feed.dates)
    split = int(len(all_dates) * 0.70)
    test_dates = all_dates[split:]

    if len(test_dates) < 2:
        raise ValueError("Holdout must contain at least two dates.")

    env = MarketEnvironment(initial_cash=INITIAL_CASH)
    restrict_dates(env, test_dates)

    agent = DQNAgent(
        state_size=STATE_SIZE,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        batch_size=64,
        memory_size=10_000,
        seed=42,
    )
    agent.load(str(MODEL_PATH))
    agent.epsilon = 0.0

    game = StackelbergGame(
        max_order_size=env.execution_engine.max_order_size
    )

    rows = []
    first_date = next(iter(env.market.values()))["date"]

    # Buy-and-hold enters at the first test day's open.
    execute_buy_and_hold(env, env.market)
    record_values(env, first_date, rows)

    while env.feed.current_step < len(env.feed.dates) - 1:
        observations = {
            agent_id: env.observe(agent_id)
            for agent_id in ("leader", "momentum", "rl_trader")
        }

        leader_actions = game.leader_action(observations["leader"])
        momentum_actions = game.follower_action(
            observations["momentum"], leader_actions
        )

        rl_actions = {}
        for symbol in env.market:
            observation = env.observe("rl_trader", symbol)
            state = encode_state(
                observation,
                initial_cash=INITIAL_CASH,
                max_order_size=env.execution_engine.max_order_size,
            )
            action = agent.act(state, training=False)
            rl_actions[symbol] = action_to_order(
                action,
                inventory=observation["inventory"],
            )

        # All policies act on current information and fill at next day's open.
        next_market = env.step()
        if next_market is None:
            break

        execute_actions(env, "leader", leader_actions, next_market)
        execute_actions(env, "momentum", momentum_actions, next_market)
        execute_actions(env, "rl_trader", rl_actions, next_market)

        date = next(iter(next_market.values()))["date"]
        record_values(env, date, rows)

    history = pd.DataFrame(rows)
    history.to_csv(HISTORY_PATH, index=False)

    metrics = []
    for agent_id, group in history.groupby("agent"):
        group = group.sort_values("date")
        values = group["portfolio_value"].astype(float)
        returns = values.pct_change().dropna()
        drawdown = values / values.cummax() - 1.0

        daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        volatility = daily_std * np.sqrt(252)
        sharpe = (
            float(returns.mean() / daily_std * np.sqrt(252))
            if daily_std > 0 else 0.0
        )

        trades = [
            record for record in env.history
            if record.get("agent") == agent_id
            and record.get("status") == "EXECUTED"
        ]

        metrics.append({
            "agent": agent_id,
            "start_date": str(group["date"].iloc[0]),
            "end_date": str(group["date"].iloc[-1]),
            "initial_value": float(values.iloc[0]),
            "final_value": float(values.iloc[-1]),
            "total_return_pct": float(
                (values.iloc[-1] / values.iloc[0] - 1.0) * 100.0
            ),
            "annualized_volatility_pct": volatility * 100.0,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": float(drawdown.min() * 100.0),
            "executed_orders": len(trades),
            "fees_paid": float(sum(float(t.get("fee", 0.0)) for t in trades)),
        })

    summary = pd.DataFrame(metrics).sort_values(
        "total_return_pct", ascending=False
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n--- HOLDOUT PERIOD STRATEGY COMPARISON ---")
    print(f"Dates: {test_dates[0]} through {test_dates[-1]}")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved portfolio history: {HISTORY_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    evaluate()
