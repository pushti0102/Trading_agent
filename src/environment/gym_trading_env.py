from typing import Dict, List, Optional, Tuple, Any, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.environment.historical_feed import HistoricalMarketFeed
from src.environment.execution_engine import ExecutionEngine
from src.agents.rl_trader import encode_state, STATE_SIZE
from src.risk.risk_overlay import PortfolioRiskOverlay
from src.environment.reward_functions import get_reward_function, BaseRewardFunction


class ObservationNormalizer:
    """Fitted strictly on training observations to eliminate lookahead data leakage."""

    def __init__(self, clip_min: float = -10.0, clip_max: float = 10.0):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.fitted = False

    def fit(self, observations: np.ndarray):
        """Fit mean and standard deviation from an array of shape (N, feature_dim)."""
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        self.mean = np.mean(obs, axis=0)
        self.std = np.std(obs, axis=0)
        self.std[self.std < 1e-8] = 1.0  # Prevent divide-by-zero on invariant features
        self.fitted = True

    def transform(self, observation: np.ndarray) -> np.ndarray:
        """Standardize observation vector using frozen training statistics."""
        obs = np.asarray(observation, dtype=np.float32)
        if not self.fitted:
            return obs
        normalized = (obs - self.mean) / self.std
        return np.clip(normalized, self.clip_min, self.clip_max).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservationNormalizer":
        norm = cls(clip_min=data.get("clip_min", -10.0), clip_max=data.get("clip_max", 10.0))
        if data.get("fitted", False) and data.get("mean") is not None:
            norm.mean = np.asarray(data["mean"], dtype=np.float64)
            norm.std = np.asarray(data["std"], dtype=np.float64)
            norm.fitted = True
        return norm


class GymTradingEnv(gym.Env):
    """Gymnasium-compatible multi-stock market trading environment.

    Supports:
    - Action modes:
        * 'continuous': Bounded vector w in [0, 1]^N representing target portfolio weights.
        * 'discrete': Discrete(3) for single-stock benchmark [0=BUY, 1=HOLD, 2=SELL].
    - Observation modes:
        * 'flattened': 1D vector (N * 10) for multi-stock MLP policies.
        * 'single_stock': 1D vector (10) for single-stock DQN benchmark.
    - Strict chronological execution:
        * Observation observed at Close(t).
        * Target weights converted into feasible integer orders executed at Open(t+1).
        * Valuation marked at Close(t+1).
    - Hard financial constraints:
        * Cash reserve floor (default 5%).
        * Maximum single-stock position concentration (default 20%).
        * Execution friction: half-spread, linear price impact, and transaction fees.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: Optional[str] = None,
        dates: Optional[List[pd.Timestamp]] = None,
        symbols: Optional[List[str]] = None,
        action_mode: str = "continuous",
        observation_mode: str = "flattened",
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_order_size: int = 100,
        impact_coefficient: float = 0.0005,
        spread: float = 0.02,
        volatility_window: int = 20,
        fundamental_window: int = 20,
        cash_reserve_fraction: float = 0.05,
        max_position_fraction: float = 0.20,
        turnover_penalty_coef: float = 0.001,
        reward_mode: str = "log_return",
        downside_penalty_coef: float = 2.0,
        risk_overlay: Optional[PortfolioRiskOverlay] = None,
        normalizer: Optional[ObservationNormalizer] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()

        self.feed = HistoricalMarketFeed(data_path=data_path)
        self.initial_cash = float(initial_cash)
        self.spread = float(spread)
        self.volatility_window = int(volatility_window)
        self.fundamental_window = int(fundamental_window)
        self.cash_reserve_fraction = float(cash_reserve_fraction)
        self.max_position_fraction = float(max_position_fraction)
        self.turnover_penalty_coef = float(turnover_penalty_coef)
        self.downside_penalty_coef = float(downside_penalty_coef)
        self.reward_mode = reward_mode
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.normalizer = normalizer

        # Risk overlay and reward function
        if risk_overlay is None:
            self.risk_overlay = PortfolioRiskOverlay(
                max_single_stock_weight=self.max_position_fraction,
                cash_reserve_fraction=self.cash_reserve_fraction,
                max_order_size=max_order_size,
            )
        else:
            self.risk_overlay = risk_overlay

        self.reward_function = get_reward_function(
            mode=self.reward_mode,
            turnover_coef=self.turnover_penalty_coef,
            downside_coef=self.downside_penalty_coef,
        )

        # Restrict dates if provided (e.g., train vs holdout)
        if dates is not None:
            self._restrict_feed_dates(dates)

        # Select stock universe
        available_symbols = sorted(self.feed.data["SYMBOL"].unique().tolist())
        if symbols is None:
            self.symbols = available_symbols
        else:
            self.symbols = [s for s in symbols if s in available_symbols]
            if not self.symbols:
                raise ValueError(f"None of requested symbols {symbols} exist in dataset")

        self.num_stocks = len(self.symbols)

        # Execution engine
        self.execution_engine = ExecutionEngine(
            transaction_cost=transaction_cost,
            max_order_size=max_order_size,
            impact_coefficient=impact_coefficient,
            spread_rate=self.spread,
        )

        # Precompute rolling features backward-only
        self._compute_features()

        # Observation and Action spaces
        self._setup_spaces()

        # State tracking
        self.cash = self.initial_cash
        self.inventory: Dict[str, int] = {s: 0 for s in self.symbols}
        self.last_prices: Dict[str, float] = {}
        self.current_market: Optional[Dict[str, Dict[str, Any]]] = None
        self.current_features: Dict[str, Dict[str, Any]] = {}
        self.portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)
        self.step_count = 0
        self.total_trades_count = 0
        self.total_fees_paid = 0.0
        self.trade_history: List[Dict[str, Any]] = []

        if seed is not None:
            self.reset(seed=seed)

    def _restrict_feed_dates(self, dates: List[Any]):
        """Filter feed to explicit chronological dates."""
        date_set = set(pd.to_datetime(d) for d in dates)
        self.feed.data = self.feed.data[self.feed.data["TIMESTAMP"].isin(date_set)].copy()
        self.feed.dates = sorted(self.feed.data["TIMESTAMP"].unique())
        self.feed.current_step = 0

    def _compute_features(self):
        """Precompute 20-day volatility and fundamental anchor with rolling window."""
        self.features_df = self.feed.data.copy().sort_values(["SYMBOL", "TIMESTAMP"])
        grouped = self.features_df.groupby("SYMBOL")

        self.features_df["VOLATILITY"] = (
            grouped["RETURN"]
            .transform(lambda s: s.rolling(self.volatility_window, min_periods=2).std())
            .fillna(0.0)
        )

        self.features_df["FUNDAMENTAL_VALUE"] = grouped["CLOSE"].transform(
            lambda s: s.rolling(self.fundamental_window, min_periods=1).mean()
        )

    def _setup_spaces(self):
        """Define Gymnasium action and observation spaces."""
        # Action space
        if self.action_mode == "continuous":
            # Target portfolio weights w_i in [0, 1] for each stock.
            # Cash weight is implicitly 1 - sum(w_i)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_stocks,),
                dtype=np.float32,
            )
        elif self.action_mode == "discrete":
            if self.num_stocks == 1:
                self.action_space = spaces.Discrete(3)  # 0: BUY, 1: HOLD, 2: SELL
            else:
                self.action_space = spaces.MultiDiscrete([3] * self.num_stocks)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # Observation space
        if self.observation_mode == "flattened":
            obs_dim = self.num_stocks * STATE_SIZE
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        elif self.observation_mode == "single_stock":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(STATE_SIZE,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.feed.reset()
        self.current_market = self.feed.get_current_market()
        if self.current_market is None:
            raise RuntimeError("Market feed is empty upon reset")

        self.last_prices = {s: float(self.current_market[s]["close"]) for s in self.symbols}
        self.cash = self.initial_cash
        self.inventory = {s: 0 for s in self.symbols}
        self.portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash
        self.high_water_mark = self.initial_cash
        self.current_drawdown = 0.0
        self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)
        self.last_risk_diagnostics = {}
        self.step_count = 0
        self.total_trades_count = 0
        self.total_fees_paid = 0.0
        self.trade_history.clear()
        self.execution_engine.records.clear()

        self._refresh_current_features()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _refresh_current_features(self):
        """Extract precomputed features for the current trading date."""
        if self.current_market is None:
            self.current_features = {}
            return
        current_date = next(iter(self.current_market.values()))["date"]
        rows = self.features_df[self.features_df["TIMESTAMP"] == current_date]
        self.current_features = rows.set_index("SYMBOL").to_dict(orient="index")

    def _get_stock_observation(self, symbol: str) -> Dict[str, Any]:
        """Compute 10 state values for a single stock."""
        data = self.current_market[symbol]
        features = self.current_features.get(symbol, {})
        vol = float(features.get("VOLATILITY", 0.0))
        fv = float(features.get("FUNDAMENTAL_VALUE", data["close"]))
        daily_ret = float(data["return"]) if data["return"] is not None else 0.0

        return {
            "price": float(data["close"]),
            "fundamental_value": fv,
            "portfolio_value": self.portfolio_value,
            "return": daily_ret,
            "volume": int(data["volume"]),
            "spread": self.spread,
            "liquidity": float(data["turnover"]),
            "volatility": vol,
            "order_imbalance": 0.0,
            "inventory": self.inventory[symbol],
            "cash": self.cash,
        }

    def _get_observation(self) -> np.ndarray:
        """Construct normalized observation vector according to observation_mode."""
        if self.observation_mode == "single_stock":
            raw_obs = self._get_stock_observation(self.symbols[0])
            state = encode_state(
                raw_obs,
                initial_cash=self.initial_cash,
                max_order_size=self.execution_engine.max_order_size,
            )
        else:
            vectors = []
            for s in self.symbols:
                raw_obs = self._get_stock_observation(s)
                vec = encode_state(
                    raw_obs,
                    initial_cash=self.initial_cash,
                    max_order_size=self.execution_engine.max_order_size,
                )
                vectors.append(vec)
            state = np.concatenate(vectors, axis=0)

        if self.normalizer is not None:
            state = self.normalizer.transform(state)

        return np.asarray(state, dtype=np.float32)

    def _project_target_weights(self, weights: np.ndarray) -> np.ndarray:
        """Project continuous raw action into feasible portfolio target weights using PortfolioRiskOverlay."""
        market_vol = 0.01
        if self.current_features:
            vols = [float(f.get("VOLATILITY", 0.01)) for f in self.current_features.values()]
            if vols:
                market_vol = float(np.mean(vols))

        feasible_w, diag = self.risk_overlay.apply(
            target_weights=weights,
            current_weights=self.current_weights,
            portfolio_value=self.portfolio_value,
            current_drawdown=self.current_drawdown,
            market_volatility=market_vol,
        )
        self.last_risk_diagnostics = diag
        return feasible_w

    def _execute_portfolio_allocation(
        self, target_weights: np.ndarray, next_market: Dict[str, Dict[str, Any]]
    ):
        """Execute portfolio rebalancing at next day's Open.

        Sells are executed first to release cash, followed by Buys.
        """
        curr_val = self.portfolio_value
        desired_values = {
            s: float(target_weights[i] * curr_val)
            for i, s in enumerate(self.symbols)
        }

        # Calculate current position values at next Open
        open_prices = {s: float(next_market[s]["open"]) for s in self.symbols}
        current_holdings = {s: self.inventory[s] for s in self.symbols}

        desired_shares = {
            s: int(desired_values[s] / max(open_prices[s], 1e-8))
            for s in self.symbols
        }

        share_deltas = {
            s: desired_shares[s] - current_holdings[s]
            for s in self.symbols
        }

        # 1. Execute SELL orders first to free up cash
        for s in self.symbols:
            delta = share_deltas[s]
            if delta < 0:
                sell_qty = min(abs(delta), current_holdings[s])
                # Obey execution engine max order size per transaction
                sell_qty = min(sell_qty, self.execution_engine.max_order_size)
                if sell_qty > 0:
                    result = self.execution_engine.execute(
                        agent_id="rl_agent",
                        symbol=s,
                        side="SELL",
                        quantity=sell_qty,
                        reference_price=open_prices[s],
                        available_cash=self.cash,
                        available_inventory=self.inventory[s],
                    )
                    if result["status"] == "EXECUTED":
                        proceeds = result["trade_value"] - result["fee"]
                        self.cash += proceeds
                        self.inventory[s] -= sell_qty
                        self.total_trades_count += 1
                        self.total_fees_paid += result["fee"]
                        self.trade_history.append(result)

        # 2. Execute BUY orders with available cash (respecting cash reserve floor)
        spendable_cash = max(0.0, self.cash - (self.initial_cash * self.cash_reserve_fraction))

        for s in self.symbols:
            delta = share_deltas[s]
            if delta > 0 and spendable_cash > 0:
                buy_qty = min(delta, self.execution_engine.max_order_size)

                # Binary search / step down to find affordable quantity under friction
                engine = self.execution_engine
                half_spread = engine.spread_rate / 2.0
                impact = engine.impact_coefficient

                def cost_for_qty(q: int) -> float:
                    adj = half_spread + impact * q
                    px = open_prices[s] * (1.0 + adj)
                    tv = px * q
                    return tv * (1.0 + engine.transaction_cost)

                while buy_qty > 0 and cost_for_qty(buy_qty) > spendable_cash:
                    buy_qty -= 1

                if buy_qty > 0:
                    result = self.execution_engine.execute(
                        agent_id="rl_agent",
                        symbol=s,
                        side="BUY",
                        quantity=buy_qty,
                        reference_price=open_prices[s],
                        available_cash=self.cash,
                        available_inventory=self.inventory[s],
                    )
                    if result["status"] == "EXECUTED":
                        cost = result["trade_value"] + result["fee"]
                        self.cash -= cost
                        self.inventory[s] += buy_qty
                        spendable_cash = max(0.0, spendable_cash - cost)
                        self.total_trades_count += 1
                        self.total_fees_paid += result["fee"]
                        self.trade_history.append(result)

    def _execute_discrete_orders(
        self, actions: Union[int, np.ndarray, List[int]], next_market: Dict[str, Dict[str, Any]]
    ):
        """Execute discrete DQN actions (0=BUY, 1=HOLD, 2=SELL)."""
        if isinstance(actions, (int, np.integer)):
            action_list = [int(actions)]
        else:
            action_list = [int(a) for a in actions]

        open_prices = {s: float(next_market[s]["open"]) for s in self.symbols}
        order_size = 5

        for i, s in enumerate(self.symbols):
            act = action_list[i] if i < len(action_list) else 1
            if act == 0:  # BUY
                spendable = max(0.0, self.cash - (self.initial_cash * self.cash_reserve_fraction))
                pos_cap = self.portfolio_value * self.max_position_fraction
                curr_pos = self.inventory[s] * open_prices[s]
                qty = min(order_size, self.execution_engine.max_order_size)
                if curr_pos + qty * open_prices[s] <= pos_cap:
                    result = self.execution_engine.execute(
                        agent_id="rl_agent",
                        symbol=s,
                        side="BUY",
                        quantity=qty,
                        reference_price=open_prices[s],
                        available_cash=spendable,
                        available_inventory=self.inventory[s],
                    )
                    if result["status"] == "EXECUTED":
                        cost = result["trade_value"] + result["fee"]
                        self.cash -= cost
                        self.inventory[s] += qty
                        self.total_trades_count += 1
                        self.total_fees_paid += result["fee"]
                        self.trade_history.append(result)
            elif act == 2:  # SELL
                if self.inventory[s] >= order_size:
                    result = self.execution_engine.execute(
                        agent_id="rl_agent",
                        symbol=s,
                        side="SELL",
                        quantity=order_size,
                        reference_price=open_prices[s],
                        available_cash=self.cash,
                        available_inventory=self.inventory[s],
                    )
                    if result["status"] == "EXECUTED":
                        proceeds = result["trade_value"] - result["fee"]
                        self.cash += proceeds
                        self.inventory[s] -= order_size
                        self.total_trades_count += 1
                        self.total_fees_paid += result["fee"]
                        self.trade_history.append(result)

    def step(
        self, action: Union[np.ndarray, int, List[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self.previous_portfolio_value = self.portfolio_value

        # Step historical feed to obtain next day Open prices
        next_market = self.feed.step()
        terminated = next_market is None
        truncated = False

        if terminated:
            # Reached end of historical dataset
            reward = 0.0
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, True, truncated, info

        # Execute orders at next Open
        prev_weights = self.current_weights.copy()
        if self.action_mode == "continuous":
            target_weights = self._project_target_weights(action)
            self._execute_portfolio_allocation(target_weights, next_market)
        else:
            self._execute_discrete_orders(action, next_market)

        # Update environment state to Close(t+1)
        self.current_market = next_market
        self.last_prices = {s: float(next_market[s]["close"]) for s in self.symbols}
        self._refresh_current_features()

        # Mark-to-market valuation at Close(t+1)
        holdings_value = sum(
            self.inventory[s] * self.last_prices[s] for s in self.symbols
        )
        self.portfolio_value = float(self.cash + holdings_value)
        self.high_water_mark = max(self.high_water_mark, self.portfolio_value)
        self.current_drawdown = float((self.portfolio_value - self.high_water_mark) / max(self.high_water_mark, 1e-8))

        # Update current realized portfolio weights
        if self.portfolio_value > 1e-8:
            self.current_weights = np.array(
                [(self.inventory[s] * self.last_prices[s]) / self.portfolio_value for s in self.symbols],
                dtype=np.float32,
            )
        else:
            self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)

        # Calculate reward using configured reward function
        reward, reward_metrics = self.reward_function.calculate(
            current_value=self.portfolio_value,
            previous_value=self.previous_portfolio_value,
            current_weights=self.current_weights,
            previous_weights=prev_weights,
            initial_capital=self.initial_cash,
        )

        ret = (self.portfolio_value / max(self.previous_portfolio_value, 1e-8)) - 1.0
        log_ret = float(np.log(max(self.portfolio_value, 1e-8) / max(self.previous_portfolio_value, 1e-8)))
        turnover = float(np.sum(np.abs(self.current_weights - prev_weights)))

        # Terminal condition check
        if self.feed.current_step >= len(self.feed.dates) - 1:
            terminated = True

        obs = self._get_observation()
        info = self._get_info()
        info["step_return"] = ret
        info["step_log_return"] = log_ret
        info["turnover"] = turnover
        info["current_drawdown"] = self.current_drawdown
        info["high_water_mark"] = self.high_water_mark
        info["risk_diagnostics"] = self.last_risk_diagnostics.copy()
        info.update(reward_metrics)

        return obs, float(reward), terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Diagnostic state information."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "inventory": self.inventory.copy(),
            "total_trades": self.total_trades_count,
            "total_fees_paid": self.total_fees_paid,
            "current_date": next(iter(self.current_market.values()))["date"] if self.current_market else None,
            "is_finished": self.feed.is_finished,
            "step_count": self.step_count,
        }
