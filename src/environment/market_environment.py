
from src.environment.historical_feed import HistoricalMarketFeed
from src.environment.execution_engine import ExecutionEngine


class MarketEnvironment:
    def __init__(
        self,
        initial_cash=100_000,
        transaction_cost=0.001,
        max_order_size=100,
        impact_coefficient=0.0005,
        spread=0.02,
        volatility_window=20,
        fundamental_window=20,
    ):
        self.feed = HistoricalMarketFeed()
        self.initial_cash = float(initial_cash)
        self.spread = float(spread)
        self.volatility_window = int(volatility_window)
        self.fundamental_window = int(fundamental_window)

        self.execution_engine = ExecutionEngine(
            transaction_cost=transaction_cost,
            max_order_size=max_order_size,
            impact_coefficient=impact_coefficient,
            spread_rate=self.spread,
        )
        # Features use only current and past observations.
        self.features = self.feed.data.copy().sort_values(
            ["SYMBOL", "TIMESTAMP"]
        )

        grouped = self.features.groupby("SYMBOL")

        self.features["VOLATILITY"] = grouped["RETURN"].transform(
            lambda s: s.rolling(
                self.volatility_window, min_periods=2
            ).std()
        ).fillna(0.0)

        self.features["FUNDAMENTAL_VALUE"] = grouped["CLOSE"].transform(
            lambda s: s.rolling(
                self.fundamental_window, min_periods=1
            ).mean()
        )

        self.reset()

    def reset(self):
        self.market = self.feed.reset()
        self.last_prices = {
            symbol: data["close"]
            for symbol, data in self.market.items()
        }

        self.agent_ids = ["leader", "rl_trader", "momentum", "buy_hold"]

        self.cash = {
            agent_id: self.initial_cash
            for agent_id in self.agent_ids
        }
        self.inventory = {
            agent_id: {symbol: 0 for symbol in self.market}
            for agent_id in self.agent_ids
        }

        self.order_imbalance = {
            symbol: 0.0 for symbol in self.market
        }
        self._net_order_quantity = {
            symbol: 0 for symbol in self.market
        }
        self._total_order_quantity = {
            symbol: 0 for symbol in self.market
        }

        self.history = []
        self.execution_engine.records.clear()
        self._refresh_features()
        return self.market

    def _refresh_features(self):
        """Load feature rows for the current simulated date."""
        if not self.market:
            self.current_features = {}
            return

        current_date = next(iter(self.market.values()))["date"]
        rows = self.features[
            self.features["TIMESTAMP"] == current_date
        ]

        self.current_features = rows.set_index("SYMBOL").to_dict(
            orient="index"
        )

    def observe(self, agent_id, symbol=None):
        """Return agent-specific observations for one or all stocks."""
        if agent_id not in self.agent_ids:
            raise ValueError(f"Unknown agent: {agent_id}")

        symbols = [symbol] if symbol is not None else list(self.market)
        states = {}

        for stock in symbols:
            if stock not in self.market:
                raise ValueError(f"Unknown stock: {stock}")

            data = self.market[stock]
            features = self.current_features[stock]
            daily_return = data["return"]

            states[stock] = {
                "price": float(data["close"]),
                "return": (
                    0.0 if daily_return is None
                    else float(daily_return)
                ),
                "volume": int(data["volume"]),
                # Simulated fractional spread, not an observed quote.
                "spread": self.spread,
                # Historical turnover used as a liquidity proxy.
                "liquidity": float(data["turnover"]),
                # Rolling standard deviation of daily returns.
                "volatility": float(features["VOLATILITY"]),
                # Net submitted quantity / total submitted quantity.
                "order_imbalance": self.order_imbalance[stock],
                "inventory": self.inventory[agent_id][stock],
                "cash": self.cash[agent_id],
                # Rolling close-price mean; a proxy, not true fundamental value.
                "fundamental_value": float(
                    features["FUNDAMENTAL_VALUE"]
                ),
            }

        portfolio_value = self.cash[agent_id] + sum(
            quantity * self.last_prices[stock]
            for stock, quantity in self.inventory[agent_id].items()
        )

        result = {
            "stocks": states,
            "cash": self.cash[agent_id],
            "inventory": self.inventory[agent_id].copy(),
            "portfolio_value": portfolio_value,
        }

        # Preserve the existing raw market field for compatibility.
        result["market"] = self.market.copy()

        if symbol is not None:
            result.update(states[symbol])

        return result

    def submit_order( self, agent_id, symbol, side, quantity, reference_price=None):
        if agent_id not in self.agent_ids:
            return {"status": "REJECTED", "reason": "Unknown agent"}

        if symbol not in self.last_prices:
            return {"status": "REJECTED", "reason": "Unknown stock"}

        result = self.execution_engine.execute(
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=(
            self.last_prices[symbol]
                if reference_price is None
                else reference_price
            ),
            available_cash=self.cash[agent_id],
            available_inventory=self.inventory[agent_id][symbol],
        )

        if result["status"] != "EXECUTED":
            return result

        qty = result["quantity"]
        price = result["execution_price"]
        fee = result["fee"]

        if result["side"] == "BUY":
            self.cash[agent_id] -= price * qty + fee
            self.inventory[agent_id][symbol] += qty
            self._net_order_quantity[symbol] += qty
        else:
            self.cash[agent_id] += price * qty - fee
            self.inventory[agent_id][symbol] -= qty
            self._net_order_quantity[symbol] -= qty

        self._total_order_quantity[symbol] += qty
        total = self._total_order_quantity[symbol]
        self.order_imbalance[symbol] = (
            self._net_order_quantity[symbol] / total
            if total else 0.0
        )

        self.history.append(result.copy())
        return result

    def step(self):
        next_market = self.feed.step()

        if next_market is None:
            self.market = None
            return None

        self.market = next_market
        self.last_prices = {
            symbol: data["close"]
            for symbol, data in next_market.items()
        }

        # Imbalance is tracked for the current simulated day only.
        self.order_imbalance = {
            symbol: 0.0 for symbol in next_market
        }
        self._net_order_quantity = {
            symbol: 0 for symbol in next_market
        }
        self._total_order_quantity = {
            symbol: 0 for symbol in next_market
        }

        self._refresh_features()
        return self.market

    def step_agents(self, actions):
        results = {}

        for agent_id, action in actions.items():
            results[agent_id] = self.submit_order(
                agent_id,
                action["symbol"],
                action["side"],
                action["quantity"],
            )

        portfolio_values = {
            agent_id: self.observe(agent_id)["portfolio_value"]
            for agent_id in self.agent_ids
        }

        next_market = self.step()

        return {
            "executions": results,
            "portfolio_values": portfolio_values,
            "next_market": next_market,
            "done": next_market is None,
        }

    @property
    def is_finished(self):
        return self.feed.is_finished
