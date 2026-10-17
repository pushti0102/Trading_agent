from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.environment.market_environment import MarketEnvironment
from src.environment.historical_feed import HistoricalMarketFeed
from src.game_theory.stackelberg_game import StackelbergGame
from src.agents.dqn_agent import DQNAgent
from src.agents.rl_trader import STATE_SIZE, action_to_order, encode_state

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "selected_stocks.csv"
MODEL_PATH = BASE_DIR / "models" / "dqn_trader.pt"
TRAIN_RESULTS_PATH = BASE_DIR / "rl_training_results.csv"
TEST_HISTORY_PATH = BASE_DIR / "test_portfolio_history.csv"
TEST_SUMMARY_PATH = BASE_DIR / "test_evaluation_summary.csv"
EPISODE_RESULTS_PATH = BASE_DIR / "episode_results.csv"

INITIAL_CASH = 100_000.0
CASH_RESERVE_FRACTION = 0.05
MAX_POSITION_FRACTION = 0.20


def calculate_affordable_quantity(env: MarketEnvironment, agent_id: str, symbol: str, side: str, requested: int, price: float) -> int:
    requested = min(int(requested), int(env.execution_engine.max_order_size))
    if requested <= 0 or price <= 0:
        return 0

    if side == "SELL":
        return min(requested, int(env.inventory[agent_id].get(symbol, 0)))

    if side != "BUY":
        return 0

    val = float(env.cash[agent_id] + sum(
        q * env.last_prices[s] for s, q in env.inventory[agent_id].items()
    ))
    held = int(env.inventory[agent_id].get(symbol, 0))
    current_price = float(env.last_prices.get(symbol, price))
    position_room = max(0.0, val * MAX_POSITION_FRACTION - held * current_price)
    max_position_qty = int(position_room / price) if price > 0 else 0

    spendable_cash = max(0.0, float(env.cash[agent_id]) - INITIAL_CASH * CASH_RESERVE_FRACTION)
    engine = env.execution_engine

    def estimated_cost(quantity: int) -> float:
        adjustment = float(engine.spread_rate) / 2.0 + float(engine.impact_coefficient) * quantity
        execution_price = price * (1.0 + adjustment)
        trade_value = execution_price * quantity
        return trade_value * (1.0 + float(engine.transaction_cost))

    quantity = min(requested, max_position_qty)
    while quantity > 0 and estimated_cost(quantity) > spendable_cash:
        quantity -= 1

    return quantity


class SimulatorService:
    def __init__(self):
        self.lock = threading.Lock()
        self.feed_data = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])
        self.feed_data = self.feed_data.sort_values(["TIMESTAMP", "SYMBOL"]).reset_index(drop=True)
        self.all_dates = sorted([str(d).split("T")[0].split()[0] for d in self.feed_data["TIMESTAMP"].unique()])
        self.symbols = sorted(self.feed_data["SYMBOL"].unique().tolist())
        
        # Split dates
        split_idx = int(len(self.all_dates) * 0.70)
        self.train_dates = self.all_dates[:split_idx]
        self.holdout_dates = self.all_dates[split_idx:]

        # Interactive simulation state
        self.interactive_env: Optional[MarketEnvironment] = None
        self.interactive_game = StackelbergGame()
        self.interactive_agent: Optional[DQNAgent] = None
        self.interactive_step_idx = 0
        self.interactive_dates: List[str] = []
        self.interactive_history_rows: List[Dict[str, Any]] = []
        self.interactive_trades: List[Dict[str, Any]] = []
        self.interactive_buy_hold_done = False

        # Background training state
        self.training_thread: Optional[threading.Thread] = None
        self.training_status = "idle"  # idle, running, completed, error, stopped
        self.training_progress = {"current_episode": 0, "total_episodes": 0, "message": ""}
        self.training_logs: List[str] = []
        self.stop_training_flag = False

        # Evaluation running flag
        self.evaluation_running = False

        # Cache of recorded evaluation trades
        self.cached_eval_trades: List[Dict[str, Any]] = []
        self._load_or_run_eval_cache()

        # Initialize interactive session on holdout dates
        self.reset_interactive("holdout")

    def _load_or_run_eval_cache(self):
        """Simulate holdout evaluation trades into memory if not already done."""
        try:
            self._generate_evaluation_trades_cache()
        except Exception as e:
            print(f"Warning: could not generate evaluation trades cache: {e}")

    def _generate_evaluation_trades_cache(self):
        """Runs holdout evaluation to collect every executed trade with timestamp."""
        env = MarketEnvironment(initial_cash=INITIAL_CASH)
        test_dates = [pd.to_datetime(d) for d in self.holdout_dates]
        env.feed.data = env.feed.data[env.feed.data["TIMESTAMP"].isin(test_dates)].copy()
        env.feed.dates = test_dates
        env.feed.current_step = 0
        env.reset()

        agent = None
        if MODEL_PATH.exists():
            agent = DQNAgent(state_size=STATE_SIZE)
            agent.load(str(MODEL_PATH))
            agent.epsilon = 0.0

        game = StackelbergGame(max_order_size=env.execution_engine.max_order_size)
        trades: List[Dict[str, Any]] = []

        # Buy & Hold first day
        first_market = env.market
        if first_market:
            first_date = str(next(iter(first_market.values()))["date"]).split("T")[0].split()[0]
            target_per_stock = INITIAL_CASH * 0.95 / len(first_market)
            for sym, data in first_market.items():
                price = float(data["open"])
                qty = min(int(env.execution_engine.max_order_size), int(target_per_stock / max(price, 1e-8)))
                while qty > 0:
                    res = env.submit_order("buy_hold", sym, "BUY", qty, price)
                    if res.get("status") == "EXECUTED":
                        trade_rec = res.copy()
                        trade_rec["date"] = first_date
                        trades.append(trade_rec)
                        break
                    if res.get("reason") != "Insufficient cash":
                        break
                    qty -= 1

        while env.feed.current_step < len(env.feed.dates) - 1:
            obs_leader = env.observe("leader")
            obs_momentum = env.observe("momentum")
            leader_actions = game.leader_action(obs_leader)
            momentum_actions = game.follower_action(obs_momentum, leader_actions)

            rl_actions = {}
            if agent:
                for symbol in env.market:
                    obs = env.observe("rl_trader", symbol)
                    state = encode_state(obs, initial_cash=INITIAL_CASH, max_order_size=env.execution_engine.max_order_size)
                    action = agent.act(state, training=False)
                    rl_actions[symbol] = action_to_order(action, inventory=obs["inventory"])

            next_market = env.step()
            if next_market is None:
                break

            current_date_str = str(next(iter(next_market.values()))["date"]).split("T")[0].split()[0]

            for agent_id, acts in [("leader", leader_actions), ("momentum", momentum_actions), ("rl_trader", rl_actions)]:
                for symbol, (side, req_qty) in acts.items():
                    if side == "HOLD" or req_qty <= 0:
                        continue
                    price = float(next_market[symbol]["open"])
                    qty = calculate_affordable_quantity(env, agent_id, symbol, side, req_qty, price)
                    if qty <= 0:
                        continue
                    res = env.submit_order(agent_id, symbol, side, qty, price)
                    if res.get("status") == "EXECUTED":
                        trade_rec = res.copy()
                        trade_rec["date"] = current_date_str
                        trades.append(trade_rec)

        self.cached_eval_trades = trades

    def get_market_stocks(self) -> List[Dict[str, Any]]:
        """Return stock overview with latest prices, daily return, 20d volatility, and turnover."""
        stocks_info = []
        for symbol in self.symbols:
            sub = self.feed_data[self.feed_data["SYMBOL"] == symbol].sort_values("TIMESTAMP")
            if sub.empty:
                continue
            last_row = sub.iloc[-1]
            prev_row = sub.iloc[-2] if len(sub) > 1 else last_row
            first_row = sub.iloc[0]

            total_return = ((last_row["CLOSE"] / first_row["CLOSE"]) - 1.0) * 100.0
            daily_returns = sub["RETURN"].dropna()
            volatility_20d = float(daily_returns.tail(20).std() * np.sqrt(252) * 100.0) if len(daily_returns) > 5 else 0.0

            stocks_info.append({
                "symbol": symbol,
                "latest_close": round(float(last_row["CLOSE"]), 2),
                "open": round(float(last_row["OPEN"]), 2),
                "high": round(float(last_row["HIGH"]), 2),
                "low": round(float(last_row["LOW"]), 2),
                "previous_close": round(float(prev_row["CLOSE"]), 2),
                "daily_return_pct": round(float(last_row["RETURN"] * 100.0) if pd.notna(last_row["RETURN"]) else 0.0, 2),
                "total_return_pct": round(total_return, 2),
                "median_volume": int(sub["TOTTRDQTY"].median()),
                "median_turnover": round(float(sub["TOTTRDVAL"].median()), 2),
                "volatility_20d_pct": round(volatility_20d, 2),
                "data_points": len(sub),
            })
        return stocks_info

    def get_market_history(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return historical OHLCV, volume, rolling volatility, and fundamental value for a stock."""
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")

        sub = self.feed_data[self.feed_data["SYMBOL"] == symbol].sort_values("TIMESTAMP").copy()
        
        # Calculate rolling 20d fundamental value and volatility
        sub["FUNDAMENTAL_VALUE"] = sub["CLOSE"].rolling(20, min_periods=1).mean()
        sub["VOLATILITY"] = sub["RETURN"].rolling(20, min_periods=2).std().fillna(0.0)

        if start_date:
            sub = sub[sub["TIMESTAMP"] >= pd.to_datetime(start_date)]
        if end_date:
            sub = sub[sub["TIMESTAMP"] <= pd.to_datetime(end_date)]

        history = []
        for _, row in sub.iterrows():
            d_str = str(row["TIMESTAMP"]).split("T")[0].split()[0]
            history.append({
                "date": d_str,
                "open": round(float(row["OPEN"]), 2),
                "high": round(float(row["HIGH"]), 2),
                "low": round(float(row["LOW"]), 2),
                "close": round(float(row["CLOSE"]), 2),
                "volume": int(row["TOTTRDQTY"]),
                "turnover": round(float(row["TOTTRDVAL"]), 2),
                "return_pct": round(float(row["RETURN"] * 100.0), 3) if pd.notna(row["RETURN"]) else 0.0,
                "volatility": round(float(row["VOLATILITY"]), 4),
                "fundamental_value": round(float(row["FUNDAMENTAL_VALUE"]), 2),
            })
        return history

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Load holdout evaluation summary metrics from test_evaluation_summary.csv."""
        if not TEST_SUMMARY_PATH.exists():
            return {"error": "Evaluation summary not yet available. Run evaluation first.", "summary": []}

        df = pd.read_csv(TEST_SUMMARY_PATH)
        records = df.to_dict(orient="records")
        for r in records:
            r["initial_value"] = round(float(r["initial_value"]), 2)
            r["final_value"] = round(float(r["final_value"]), 2)
            r["total_return_pct"] = round(float(r["total_return_pct"]), 2)
            r["annualized_volatility_pct"] = round(float(r["annualized_volatility_pct"]), 2)
            r["sharpe_ratio"] = round(float(r["sharpe_ratio"]), 3)
            r["max_drawdown_pct"] = round(float(r["max_drawdown_pct"]), 2)
            r["fees_paid"] = round(float(r["fees_paid"]), 2)
            r["start_date"] = str(r["start_date"]).split("T")[0].split()[0]
            r["end_date"] = str(r["end_date"]).split("T")[0].split()[0]

        return {
            "split": "holdout_test",
            "test_dates_count": len(self.holdout_dates),
            "start_date": self.holdout_dates[0],
            "end_date": self.holdout_dates[-1],
            "initial_cash": INITIAL_CASH,
            "assumptions": {
                "transaction_cost_pct": 0.1,
                "quoted_spread_pct": 2.0,
                "impact_coefficient": 0.0005,
                "max_order_size": 100,
                "cash_reserve_pct": 5.0,
                "max_position_pct": 20.0
            },
            "summary": records
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Return daily portfolio values for each agent formatted for charting."""
        if not TEST_HISTORY_PATH.exists():
            return []

        df = pd.read_csv(TEST_HISTORY_PATH)
        df["date"] = df["date"].astype(str).apply(lambda d: d.split("T")[0].split()[0])
        
        # Pivot table by date with agent columns
        pivot = df.pivot(index="date", columns="agent", values="portfolio_value").reset_index()
        pivot = pivot.sort_values("date")
        
        results = []
        for _, row in pivot.iterrows():
            item = {"date": row["date"]}
            for col in ["rl_trader", "leader", "momentum", "buy_hold"]:
                if col in row and pd.notna(row[col]):
                    item[col] = round(float(row[col]), 2)
            results.append(item)
        return results

    def get_training_results(self) -> Dict[str, Any]:
        """Return episode-by-episode RL training metrics."""
        history = []
        if TRAIN_RESULTS_PATH.exists():
            df = pd.read_csv(TRAIN_RESULTS_PATH)
            for _, r in df.iterrows():
                history.append({
                    "episode": int(r["episode"]),
                    "final_portfolio_value": round(float(r["final_portfolio_value"]), 2),
                    "total_reward": round(float(r["total_reward"]), 4),
                    "mean_loss": round(float(r["mean_loss"]), 7),
                    "executed_orders": int(r["executed_orders"]),
                    "rejected_orders": int(r["rejected_orders"]),
                    "return_pct": round(float(r["return_pct"]), 2),
                    "epsilon": round(float(r["epsilon"]), 3),
                })

        model_exists = MODEL_PATH.exists()
        model_size = MODEL_PATH.stat().st_size if model_exists else 0
        model_modified = time.ctime(MODEL_PATH.stat().st_mtime) if model_exists else None

        return {
            "status": self.training_status,
            "progress": self.training_progress,
            "logs": self.training_logs[-50:],
            "model": {
                "exists": model_exists,
                "path": str(MODEL_PATH.name),
                "size_bytes": model_size,
                "last_modified": model_modified,
            },
            "episodes": history,
            "train_dates_count": len(self.train_dates),
            "train_start_date": self.train_dates[0],
            "train_end_date": self.train_dates[-1]
        }

    def start_training(self, episodes: int = 10):
        """Trigger RL training in background thread safely."""
        with self.lock:
            if self.training_status == "running":
                raise ValueError("Training is already currently in progress.")
            self.training_status = "running"
            self.stop_training_flag = False
            self.training_progress = {"current_episode": 0, "total_episodes": episodes, "message": "Starting training..."}
            self.training_logs = [f"Initialized training session with {episodes} episodes on training split ({self.train_dates[0]} - {self.train_dates[-1]})."]

        def worker():
            try:
                from train_rl import run_training_episode, restrict_dates
                train_env = MarketEnvironment(initial_cash=INITIAL_CASH)
                restrict_dates(train_env, [pd.to_datetime(d) for d in self.train_dates])

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
                for ep in range(1, episodes + 1):
                    if self.stop_training_flag:
                        self.training_status = "stopped"
                        self.training_logs.append("Training halted by user request.")
                        return

                    self.training_progress["current_episode"] = ep
                    self.training_progress["message"] = f"Simulating Episode {ep}/{episodes}..."
                    
                    result = run_training_episode(train_env, agent)
                    ret_pct = (result["final_portfolio_value"] / INITIAL_CASH - 1.0) * 100.0

                    row = {
                        "episode": ep,
                        **result,
                        "return_pct": ret_pct,
                        "epsilon": agent.epsilon,
                    }
                    rows.append(row)

                    log_msg = (
                        f"Episode {ep}/{episodes} | Return: {ret_pct:.2f}% | "
                        f"Final Value: ₹{result['final_portfolio_value']:.2f} | "
                        f"Orders: {result['executed_orders']} | Loss: {result['mean_loss']:.6f} | Epsilon: {agent.epsilon:.3f}"
                    )
                    self.training_logs.append(log_msg)

                # Save model & training results
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(MODEL_PATH))
                pd.DataFrame(rows).to_csv(TRAIN_RESULTS_PATH, index=False)
                
                self.training_status = "completed"
                self.training_progress["message"] = "Training successfully completed. Model saved."
                self.training_logs.append(f"Model saved to {MODEL_PATH}. rl_training_results.csv updated.")
            except Exception as e:
                self.training_status = "error"
                self.training_progress["message"] = f"Training failed: {str(e)}"
                self.training_logs.append(f"ERROR: {str(e)}")

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Signal training worker to stop."""
        if self.training_status == "running":
            self.stop_training_flag = True
            self.training_logs.append("Stop request received, finishing current episode...")

    def run_evaluation(self) -> Dict[str, Any]:
        """Run holdout evaluation and update summary CSVs."""
        if self.evaluation_running:
            raise ValueError("Evaluation is already running.")
        if not MODEL_PATH.exists():
            raise ValueError("No trained model found. Please train or provide models/dqn_trader.pt first.")

        self.evaluation_running = True
        try:
            import evaluate_test
            evaluate_test.evaluate()
            self._generate_evaluation_trades_cache()
            return self.get_evaluation_summary()
        finally:
            self.evaluation_running = False

    def get_trades(
        self,
        agent: Optional[str] = None,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """Filter and paginate recorded trades."""
        trades = list(self.cached_eval_trades)
        # If interactive trades exist, prepend them
        if self.interactive_trades:
            trades = self.interactive_trades + trades

        filtered = []
        for t in trades:
            if agent and agent.lower() != "all" and t.get("agent") != agent:
                continue
            if symbol and symbol.upper() != "ALL" and t.get("symbol") != symbol:
                continue
            if side and side.upper() != "ALL" and t.get("side") != side:
                continue
            if status and status.upper() != "ALL" and t.get("status") != status:
                continue
            if search:
                term = search.lower()
                matches = (
                    term in str(t.get("symbol", "")).lower()
                    or term in str(t.get("agent", "")).lower()
                    or term in str(t.get("date", "")).lower()
                    or term in str(t.get("side", "")).lower()
                )
                if not matches:
                    continue
            filtered.append(t)

        total_count = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        items = filtered[start_idx:end_idx]

        for item in items:
            for k in ["reference_price", "execution_price", "trade_value", "fee", "spread_rate", "impact_rate"]:
                if k in item and item[k] is not None:
                    item[k] = round(float(item[k]), 4)

        return {
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total_count + page_size - 1) // page_size),
            "items": items
        }

    def get_stackelberg_game_analysis(self) -> Dict[str, Any]:
        """Extract empirical leader-follower joint action distribution and actual decisions."""
        records = []
        action_matrix = {
            "BUY_BUY": 0, "BUY_HOLD": 0, "BUY_SELL": 0,
            "HOLD_BUY": 0, "HOLD_HOLD": 0, "HOLD_SELL": 0,
            "SELL_BUY": 0, "SELL_HOLD": 0, "SELL_SELL": 0,
        }

        # Use the first 20 dates of holdout to generate detailed step logs
        env = MarketEnvironment(initial_cash=INITIAL_CASH)
        test_dates = [pd.to_datetime(d) for d in self.holdout_dates[:25]]
        env.feed.data = env.feed.data[env.feed.data["TIMESTAMP"].isin(test_dates)].copy()
        env.feed.dates = test_dates
        env.feed.current_step = 0
        env.reset()
        game = StackelbergGame()

        while env.feed.current_step < len(env.feed.dates) - 1:
            obs_leader = env.observe("leader")
            obs_follower = env.observe("momentum")
            leader_acts = game.leader_action(obs_leader)
            follower_acts = game.follower_action(obs_follower, leader_acts)

            next_m = env.step()
            if next_m is None:
                break
            d_str = str(next(iter(next_m.values()))["date"]).split("T")[0].split()[0]

            for sym in env.market:
                l_side, l_qty = leader_acts[sym]
                f_side, f_qty = follower_acts[sym]
                pair_key = f"{l_side}_{f_side}"
                if pair_key in action_matrix:
                    action_matrix[pair_key] += 1

                st = obs_leader["stocks"][sym]
                records.append({
                    "date": d_str,
                    "symbol": sym,
                    "price": round(float(st["price"]), 2),
                    "fundamental_value": round(float(st["fundamental_value"]), 2),
                    "valuation_gap_pct": round(((float(st["price"]) / max(float(st["fundamental_value"]), 1e-6)) - 1.0) * 100.0, 2),
                    "market_return_pct": round(float(st["return"]) * 100.0, 2),
                    "leader_action": l_side,
                    "leader_quantity": l_qty,
                    "follower_action": f_side,
                    "follower_quantity": f_qty,
                    "leader_rationale": "Undervalued (< -1%)" if l_side == "BUY" else ("Overvalued (> +1%)" if l_side == "SELL" else "Fair Range [±1%]"),
                    "follower_rationale": "Followed Leader BUY with positive return" if (l_side == "BUY" and f_side == "BUY") else (
                        "Followed Leader SELL" if (l_side == "SELL" and f_side == "SELL") else (
                            "Positive momentum breakout (> +1%)" if f_side == "BUY" else (
                                "Negative momentum plunge (< -1%)" if f_side == "SELL" else "Neutral / No trend"
                            )
                        )
                    )
                })

        return {
            "game_type": "One-step Sequential Stackelberg Game with Execution Friction",
            "leader_profile": {
                "name": "Stackelberg Leader",
                "role": "First-mover Informed Trader",
                "objective": "Mean-reversion exploiting 20-day rolling fundamental anchor",
                "rules": "Submits BUY if Price < 0.99 * FV; SELL if Price > 1.01 * FV; otherwise HOLD."
            },
            "follower_profile": {
                "name": "Momentum Follower",
                "role": "Second-mover Reactive Trader",
                "objective": "Trend following and leader signaling response",
                "rules": "Follows leader BUY if 1-day Return >= 0; follows leader SELL if held; otherwise reacts to ±1% price momentum."
            },
            "action_matrix": action_matrix,
            "sample_decisions": records[:30]
        }

    # Interactive Simulation Methods
    def reset_interactive(self, split: str = "holdout") -> Dict[str, Any]:
        """Reset the interactive stepping environment."""
        with self.lock:
            self.interactive_env = MarketEnvironment(initial_cash=INITIAL_CASH)
            if split == "holdout":
                dates = [pd.to_datetime(d) for d in self.holdout_dates]
            elif split == "train":
                dates = [pd.to_datetime(d) for d in self.train_dates]
            else:
                dates = [pd.to_datetime(d) for d in self.all_dates]

            self.interactive_env.feed.data = self.interactive_env.feed.data[
                self.interactive_env.feed.data["TIMESTAMP"].isin(dates)
            ].copy()
            self.interactive_env.feed.dates = dates
            self.interactive_env.feed.current_step = 0
            self.interactive_env.reset()

            self.interactive_dates = [str(d).split("T")[0].split()[0] for d in dates]
            self.interactive_step_idx = 0
            self.interactive_history_rows = []
            self.interactive_trades = []
            self.interactive_buy_hold_done = False

            if MODEL_PATH.exists():
                self.interactive_agent = DQNAgent(state_size=STATE_SIZE)
                self.interactive_agent.load(str(MODEL_PATH))
                self.interactive_agent.epsilon = 0.0
            else:
                self.interactive_agent = None

            first_date = str(next(iter(self.interactive_env.market.values()))["date"]).split("T")[0].split()[0]
            self._record_interactive_step(first_date)

            return self.get_interactive_state()

    def _record_interactive_step(self, date: str):
        env = self.interactive_env
        row = {"date": date}
        for agent_id in ("rl_trader", "leader", "momentum", "buy_hold"):
            obs = env.observe(agent_id)
            row[f"{agent_id}_value"] = round(float(obs["portfolio_value"]), 2)
            row[f"{agent_id}_cash"] = round(float(obs["cash"]), 2)
        self.interactive_history_rows.append(row)

    def step_interactive(self) -> Dict[str, Any]:
        """Step one day forward in the interactive simulation."""
        with self.lock:
            env = self.interactive_env
            if not env or env.feed.current_step >= len(env.feed.dates) - 1:
                return {"is_finished": True, "state": self.get_interactive_state()}

            # First day buy and hold if not done
            if not self.interactive_buy_hold_done:
                first_market = env.market
                target_per_stock = INITIAL_CASH * 0.95 / len(first_market)
                first_date_str = str(next(iter(first_market.values()))["date"]).split("T")[0].split()[0]
                for sym, data in first_market.items():
                    price = float(data["open"])
                    qty = min(int(env.execution_engine.max_order_size), int(target_per_stock / max(price, 1e-8)))
                    while qty > 0:
                        res = env.submit_order("buy_hold", sym, "BUY", qty, price)
                        if res.get("status") == "EXECUTED":
                            res_copy = res.copy()
                            res_copy["date"] = first_date_str
                            self.interactive_trades.insert(0, res_copy)
                            break
                        if res.get("reason") != "Insufficient cash":
                            break
                        qty -= 1
                self.interactive_buy_hold_done = True

            obs_leader = env.observe("leader")
            obs_momentum = env.observe("momentum")
            leader_actions = self.interactive_game.leader_action(obs_leader)
            momentum_actions = self.interactive_game.follower_action(obs_momentum, leader_actions)

            rl_actions = {}
            if self.interactive_agent:
                for symbol in env.market:
                    obs = env.observe("rl_trader", symbol)
                    state = encode_state(obs, initial_cash=INITIAL_CASH, max_order_size=env.execution_engine.max_order_size)
                    act = self.interactive_agent.act(state, training=False)
                    rl_actions[symbol] = action_to_order(act, inventory=obs["inventory"])

            next_market = env.step()
            if next_market is None:
                return {"is_finished": True, "state": self.get_interactive_state()}

            current_date_str = str(next(iter(next_market.values()))["date"]).split("T")[0].split()[0]
            step_executions = []

            for agent_id, acts in [("leader", leader_actions), ("momentum", momentum_actions), ("rl_trader", rl_actions)]:
                for symbol, (side, req_qty) in acts.items():
                    if side == "HOLD" or req_qty <= 0:
                        continue
                    price = float(next_market[symbol]["open"])
                    qty = calculate_affordable_quantity(env, agent_id, symbol, side, req_qty, price)
                    if qty <= 0:
                        continue
                    res = env.submit_order(agent_id, symbol, side, qty, price)
                    if res.get("status") == "EXECUTED":
                        trade_rec = res.copy()
                        trade_rec["date"] = current_date_str
                        self.interactive_trades.insert(0, trade_rec)
                        step_executions.append(trade_rec)

            self.interactive_step_idx += 1
            self._record_interactive_step(current_date_str)

            return {
                "is_finished": env.feed.current_step >= len(env.feed.dates) - 1,
                "current_date": current_date_str,
                "step_executions": step_executions,
                "state": self.get_interactive_state()
            }

    def get_interactive_state(self) -> Dict[str, Any]:
        """Return snapshot of current interactive simulation."""
        env = self.interactive_env
        if not env or not env.market:
            return {
                "active": False,
                "current_step": 0,
                "total_steps": len(self.interactive_dates),
                "current_date": self.interactive_dates[0] if self.interactive_dates else "",
                "agents": {},
                "market_prices": {},
                "history": []
            }

        curr_date = str(next(iter(env.market.values()))["date"]).split("T")[0].split()[0]
        agents_data = {}
        for agent_id in ("rl_trader", "leader", "momentum", "buy_hold"):
            obs = env.observe(agent_id)
            val = float(obs["portfolio_value"])
            agents_data[agent_id] = {
                "cash": round(float(obs["cash"]), 2),
                "portfolio_value": round(val, 2),
                "return_pct": round(((val / INITIAL_CASH) - 1.0) * 100.0, 2),
                "inventory": {s: int(qty) for s, qty in obs["inventory"].items() if qty > 0}
            }

        prices = {s: round(float(data["close"]), 2) for s, data in env.market.items()}

        return {
            "active": True,
            "current_step": self.interactive_step_idx,
            "total_steps": len(self.interactive_dates),
            "current_date": curr_date,
            "is_finished": env.feed.current_step >= len(env.feed.dates) - 1,
            "agents": agents_data,
            "market_prices": prices,
            "history": self.interactive_history_rows[-60:]
        }


# Singleton service instance
simulator_service = SimulatorService()
