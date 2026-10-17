
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DQNAgent
from src.agents.rl_trader import STATE_SIZE, action_to_order, encode_state
from src.environment.market_environment import MarketEnvironment


EPISODES = 10
INITIAL_CASH = 100_000.0
AGENT_ID = "rl_trader"
MODEL_PATH = Path("models/dqn_trader.pt")
RESULTS_PATH = Path("rl_training_results.csv")

CASH_RESERVE_FRACTION = 0.05
MAX_POSITION_FRACTION = 0.20
REJECTED_ORDER_PENALTY = 1.0


def restrict_dates(env, dates):
    """Restrict replay to a chronological date range."""
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
    """Limit order size using available cash and per-stock exposure."""
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
    position_value = held * current_price
    position_room = max(
        0.0,
        value * MAX_POSITION_FRACTION - position_value,
    )
    max_position_qty = int(position_room / price)

    cash_reserve = INITIAL_CASH * CASH_RESERVE_FRACTION
    spendable_cash = max(
        0.0,
        float(env.cash[agent_id]) - cash_reserve,
    )

    engine = env.execution_engine
    half_spread = float(engine.spread_rate) / 2.0
    impact = float(engine.impact_coefficient)

    def estimated_cost(quantity):
        adjustment = half_spread + impact * quantity
        execution_price = price * (1.0 + adjustment)
        trade_value = execution_price * quantity
        fee = trade_value * float(engine.transaction_cost)
        return trade_value + fee

    quantity = min(requested, max_position_qty)
    while quantity > 0 and estimated_cost(quantity) > spendable_cash:
        quantity -= 1

    return quantity


def run_training_episode(env, agent):
    env.reset()

    total_reward = 0.0
    losses = []
    rejected_orders = 0
    executed_orders = 0

    # Observe close(t), execute at open(t+1), mark value at close(t+1).
    while env.feed.current_step < len(env.feed.dates) - 1:
        observations = {
            symbol: env.observe(AGENT_ID, symbol)
            for symbol in env.market
        }

        states = {
            symbol: encode_state(
                observation,
                initial_cash=INITIAL_CASH,
                max_order_size=env.execution_engine.max_order_size,
            )
            for symbol, observation in observations.items()
        }

        actions = {
            symbol: agent.act(state, training=True)
            for symbol, state in states.items()
        }

        old_prices = {
            symbol: float(observations[symbol]["price"])
            for symbol in observations
        }
        old_inventory = {
            symbol: int(observations[symbol]["inventory"])
            for symbol in observations
        }

        next_market = env.step()
        if next_market is None:
            break

        done = env.feed.current_step == len(env.feed.dates) - 1
        next_open = {
            symbol: float(data["open"])
            for symbol, data in next_market.items()
        }
        results = {}

        # Execute each stock's action with cash and position constraints.
        for symbol, action in actions.items():
            side, requested = action_to_order(
                action,
                inventory=old_inventory[symbol],
            )

            if side == "HOLD" or requested <= 0:
                results[symbol] = {
                    "status": "HOLD",
                    "action": 1,
                }
                continue

            quantity = affordable_quantity(
                env,
                AGENT_ID,
                symbol,
                side,
                requested,
                next_open[symbol],
            )

            if quantity <= 0:
                results[symbol] = {
                    "status": "SKIPPED",
                    "action": action,
                }
                continue

            result = env.submit_order(
                agent_id=AGENT_ID,
                symbol=symbol,
                side=side,
                quantity=quantity,
                reference_price=next_open[symbol],
            )
            result["action"] = action
            results[symbol] = result

            if result["status"] == "EXECUTED":
                executed_orders += 1
            elif result["status"] == "REJECTED":
                rejected_orders += 1

        # Reward is the change in marked portfolio value attributable
        # to each stock, net of trading fees and any rejection penalty.
        for symbol, state in states.items():
            close_price = float(next_market[symbol]["close"])
            prior_close = old_prices[symbol]
            old_qty = old_inventory[symbol]

            result = results[symbol]
            new_qty = int(env.inventory[AGENT_ID].get(symbol, 0))

            reward_dollars = (
                old_qty * (close_price - prior_close)
            )

            # Add P&L for inventory acquired/sold at the next open.
            if result["status"] == "EXECUTED":
                signed_qty = (
                    result["quantity"]
                    if result["side"] == "BUY"
                    else -result["quantity"]
                )
                reward_dollars += signed_qty * (
                    close_price - result["execution_price"]
                )
                reward_dollars -= float(result["fee"])

            elif result["status"] == "REJECTED":
                reward_dollars -= REJECTED_ORDER_PENALTY

            reward = reward_dollars / INITIAL_CASH
            total_reward += reward

            next_observation = env.observe(AGENT_ID, symbol)
            next_state = encode_state(
                next_observation,
                initial_cash=INITIAL_CASH,
                max_order_size=env.execution_engine.max_order_size,
            )

            agent.remember(
                state=state,
                action=int(result["action"]),
                reward=float(reward),
                next_state=next_state,
                done=done,
            )

        loss = agent.train_step()
        if loss is not None:
            losses.append(float(loss))

        if done:
            break

    return {
        "final_portfolio_value": portfolio_value(env, AGENT_ID),
        "total_reward": total_reward,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "executed_orders": executed_orders,
        "rejected_orders": rejected_orders,
    }


def main():
    reference_env = MarketEnvironment(initial_cash=INITIAL_CASH)
    all_dates = list(reference_env.feed.dates)

    if len(all_dates) < 10:
        raise ValueError(f"Need at least 10 dates; found {len(all_dates)}")

    split = int(len(all_dates) * 0.70)
    train_dates = all_dates[:split]
    holdout_dates = all_dates[split:]

    print(f"Total dates: {len(all_dates)}")
    print(
        f"Training: {len(train_dates)} dates "
        f"({train_dates[0]} through {train_dates[-1]})"
    )
    print(
        f"Reserved holdout: {len(holdout_dates)} dates "
        f"({holdout_dates[0]} through {holdout_dates[-1]})"
    )

    train_env = MarketEnvironment(initial_cash=INITIAL_CASH)
    restrict_dates(train_env, train_dates)

    agent = DQNAgent(
        state_size=STATE_SIZE,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10_000,
        seed=42,
    )

    rows = []

    for episode in range(1, EPISODES + 1):
        result = run_training_episode(train_env, agent)

        row = {
            "episode": episode,
            **result,
            "return_pct": (
                result["final_portfolio_value"] / INITIAL_CASH - 1.0
            ) * 100.0,
            "epsilon": agent.epsilon,
        }
        rows.append(row)

        print(
            f"Train {episode}/{EPISODES} | "
            f"Return {row['return_pct']:.2f}% | "
            f"Value {row['final_portfolio_value']:.2f} | "
            f"Orders {row['executed_orders']} | "
            f"Rejected {row['rejected_orders']} | "
            f"Epsilon {agent.epsilon:.3f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(MODEL_PATH))
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved training log: {RESULTS_PATH}")
    print("Training complete. Run python evaluate_test.py.")


if __name__ == "__main__":
    main()
