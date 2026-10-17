
import pandas as pd

from src.environment.market_environment import MarketEnvironment
from src.game_theory.stackelberg_game import StackelbergGame


AGENTS = ("leader", "momentum", "buy_hold")
CASH_RESERVE = 0.05
MAX_POSITION_FRACTION = 0.20


def affordable_quantity(env, agent_id, symbol, side, requested, price):
    """Limit order quantity using cash, costs, and position limits."""
    if requested <= 0 or price <= 0:
        return 0

    engine = env.execution_engine
    requested = min(int(requested), engine.max_order_size)

    if side == "SELL":
        held = env.inventory[agent_id][symbol]
        return min(requested, held)

    if side != "BUY":
        return 0

    cash = env.cash[agent_id]
    spendable_cash = max(0.0, cash - env.initial_cash * CASH_RESERVE)

    # Include the engine's spread, impact, and transaction fee.
    def estimated_cost(quantity):
        impact = engine.impact_coefficient * quantity
        execution_price = price * (
            1 + engine.spread_rate / 2 + impact
        )
        return execution_price * quantity * (
            1 + engine.transaction_cost
        )

    quantity = 0
    for candidate in range(1, requested + 1):
        if estimated_cost(candidate) <= spendable_cash:
            quantity = candidate
        else:
            break

    # Avoid concentrating too much of the portfolio in one stock.
    portfolio_value = env.observe(agent_id)["portfolio_value"]
    current_value = (
        env.inventory[agent_id][symbol] * env.last_prices[symbol]
    )
    position_cap = portfolio_value * MAX_POSITION_FRACTION
    room = max(0.0, position_cap - current_value)

    while quantity > 0 and quantity * price > room:
        quantity -= 1

    return quantity


def execute_actions(env, agent_id, actions, market):
    for symbol, (side, requested_quantity) in actions.items():
        side = str(side).upper()

        if side == "HOLD" or requested_quantity <= 0:
            continue

        price = float(market[symbol]["open"])

        quantity = affordable_quantity(
            env=env,
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            requested=requested_quantity,
            price=price,
        )

        if quantity <= 0:
            continue

        result = env.submit_order(
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=price,
        )

        if result["status"] == "REJECTED":
            print(
                f"Rejected: {agent_id} {symbol}: "
                f"{result['reason']}"
            )


def execute_buy_and_hold(env, market):
    """Invest approximately equal amounts, then hold."""
    symbols = list(market)
    if not symbols:
        return

    target_per_stock = (
        env.initial_cash * (1 - CASH_RESERVE) / len(symbols)
    )

    for symbol, data in market.items():
        price = float(data["open"])
        quantity = min(
            env.execution_engine.max_order_size,
            int(target_per_stock / max(price, 1e-8)),
        )

        quantity = affordable_quantity(
            env, "buy_hold", symbol, "BUY", quantity, price
        )

        if quantity <= 0:
            continue

        result = env.submit_order(
            agent_id="buy_hold",
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            reference_price=price,
        )

        if result["status"] == "REJECTED":
            print(
                f"Rejected: buy_hold {symbol}: "
                f"{result['reason']}"
            )


def record_portfolios(env, date, records):
    for agent_id in AGENTS:
        observation = env.observe(agent_id)
        records.append({
            "date": date,
            "agent": agent_id,
            "cash": observation["cash"],
            "holdings_value": (
                observation["portfolio_value"] - observation["cash"]
            ),
            "portfolio_value": observation["portfolio_value"],
        })


def main():
    env = MarketEnvironment()
    game = StackelbergGame()
    market = env.reset()

    records = []
    buy_hold_initialized = False

    current_date = next(iter(market.values()))["date"]
    record_portfolios(env, current_date, records)

    while not env.is_finished:
        observations = {
            agent_id: env.observe(agent_id)
            for agent_id in env.agent_ids
        }

        leader_actions = game.leader_action(observations["leader"])

        follower_actions = game.follower_action(
            observations["momentum"],
            leader_actions,
        )

        next_market = env.step()
        if next_market is None:
            break

        execute_actions(
            env, "leader", leader_actions, next_market
        )
        execute_actions(
            env, "momentum", follower_actions, next_market
        )

        if not buy_hold_initialized:
            execute_buy_and_hold(env, next_market)
            buy_hold_initialized = True

        current_date = next(iter(next_market.values()))["date"]
        record_portfolios(env, current_date, records)

    results = pd.DataFrame(records)
    results.to_csv("episode_results.csv", index=False)

    final_values = (
        results.sort_values("date")
        .groupby("agent")
        .tail(1)
        .set_index("agent")["portfolio_value"]
        .sort_values(ascending=False)
    )

    print("\nEpisode completed.")
    print(f"Trading days: {results['date'].nunique()}")
    print("\nFinal portfolio values:")
    print(final_values.round(2))
    print("\nExecution counts:")

    for agent_id in AGENTS:
        count = sum(
            trade["agent"] == agent_id
            for trade in env.history
        )
        print(f"{agent_id}: {count}")

    print("\nSaved: episode_results.csv")


if __name__ == "__main__":
    main()
