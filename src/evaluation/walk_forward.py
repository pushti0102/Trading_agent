from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.environment.gym_trading_env import GymTradingEnv, ObservationNormalizer
from src.risk.risk_overlay import PortfolioRiskOverlay
from src.analytics.financial_metrics import FinancialMetricsCalculator
from src.agents.rule_based_agents import (
    MeanReversionAgent,
    VolatilityTargetingAgent,
    RiskParityAgent,
    StackelbergBestResponseAgent,
    StackelbergLeaderAgent,
)
from src.agents.continuous_rl_agents import create_ppo_agent, create_a2c_agent
from src.agents.ensemble_agents import ValidationWeightedEnsemble, EqualWeightEnsemble


class WalkForwardSplitter:
    """Chronological Expanding-Window Walk-Forward Splitter."""

    def __init__(self, all_dates: List[pd.Timestamp], min_train_days: int = 200, test_days: int = 70):
        self.all_dates = sorted(all_dates)
        self.min_train_days = int(min_train_days)
        self.test_days = int(test_days)

    def generate_folds(self) -> List[Dict[str, Any]]:
        """Yield expanding training windows followed strictly by out-of-sample test windows."""
        folds = []
        n_total = len(self.all_dates)
        current_train_end = self.min_train_days
        fold_idx = 1

        while current_train_end + self.test_days <= n_total:
            train_dates = self.all_dates[:current_train_end]
            test_dates = self.all_dates[current_train_end : current_train_end + self.test_days]

            folds.append({
                "fold": fold_idx,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "train_start": str(train_dates[0])[:10],
                "train_end": str(train_dates[-1])[:10],
                "test_start": str(test_dates[0])[:10],
                "test_end": str(test_dates[-1])[:10],
                "train_count": len(train_dates),
                "test_count": len(test_dates),
            })

            fold_idx += 1
            current_train_end += self.test_days

        # Include remaining dates in the last fold if any
        if current_train_end < n_total:
            train_dates = self.all_dates[:current_train_end]
            test_dates = self.all_dates[current_train_end:]
            folds.append({
                "fold": fold_idx,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "train_start": str(train_dates[0])[:10],
                "train_end": str(train_dates[-1])[:10],
                "test_start": str(test_dates[0])[:10],
                "test_end": str(test_dates[-1])[:10],
                "train_count": len(train_dates),
                "test_count": len(test_dates),
            })

        return folds


class WalkForwardEvaluator:
    """Orchestrates multi-fold walk-forward validation across strategies and seeds."""

    def __init__(
        self,
        data_path: str = "data/processed/selected_stocks.csv",
        risk_free_rate: float = 0.05,
        min_train_days: int = 220,
        test_days: int = 70,
        seeds: Optional[List[int]] = None,
    ):
        self.data_path = data_path
        self.df = pd.read_csv(data_path, parse_dates=["TIMESTAMP"])
        self.all_dates = sorted(self.df["TIMESTAMP"].unique())
        self.splitter = WalkForwardSplitter(self.all_dates, min_train_days=min_train_days, test_days=test_days)
        self.calculator = FinancialMetricsCalculator(risk_free_rate=risk_free_rate)
        self.seeds = seeds or [42, 101]

    def run_walk_forward(self, output_dir: str = "results/olive_trade_v2/phase_06_walk_forward") -> pd.DataFrame:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        folds = self.splitter.generate_folds()
        print(f"Generated {len(folds)} chronological walk-forward folds.")

        results_records = []
        fold_summaries = []

        risk_overlay = PortfolioRiskOverlay(
            max_single_stock_weight=0.20,
            cash_reserve_fraction=0.05,
            max_turnover_per_step=0.25,
            volatility_target=0.12,
            max_drawdown_limit=0.10,
        )

        for fold in folds:
            f_num = fold["fold"]
            print(f"\n==========================================")
            print(f"Executing Walk-Forward Fold {f_num}/{len(folds)}")
            print(f"Train: {fold['train_start']} to {fold['train_end']} ({fold['train_count']} days)")
            print(f"Test:  {fold['test_start']} to {fold['test_end']} ({fold['test_count']} days)")
            print(f"==========================================")

            # 1. Fit normalizer strictly on this fold's training partition
            temp_env = GymTradingEnv(
                dates=fold["train_dates"],
                action_mode="continuous",
                observation_mode="flattened",
            )
            obs_samples = []
            obs, _ = temp_env.reset(seed=42)
            obs_samples.append(obs)
            done = False
            while not done:
                act = np.random.uniform(0.01, 0.04, size=temp_env.num_stocks).astype(np.float32)
                obs, rew, term, trunc, _ = temp_env.step(act)
                obs_samples.append(obs)
                done = term or trunc

            normalizer = ObservationNormalizer(clip_min=-5.0, clip_max=5.0)
            normalizer.fit(np.array(obs_samples))

            # 2. Instantiate and evaluate models for this fold
            # Rule-based and game-theoretic models
            strategies = [
                ("Volatility Targeting", VolatilityTargetingAgent(num_stocks=40, target_annual_vol=0.12), "continuous"),
                ("Risk Parity", RiskParityAgent(num_stocks=40), "continuous"),
                ("Mean Reversion", MeanReversionAgent(num_stocks=40), "continuous"),
                ("Stackelberg Best Response", StackelbergBestResponseAgent(num_stocks=40), "discrete"),
                ("Stackelberg Leader", StackelbergLeaderAgent(num_stocks=40), "discrete"),
            ]

            # Compute benchmark equity for this fold
            bench_env = GymTradingEnv(
                dates=fold["test_dates"],
                action_mode="continuous",
                observation_mode="flattened",
                normalizer=normalizer,
                risk_overlay=risk_overlay,
                initial_cash=100000.0,
                seed=42,
            )
            obs, info = bench_env.reset(seed=42)
            bench_equity = [100000.0]
            first_day = True
            done = False
            while not done:
                if first_day:
                    w = np.full(40, 0.90 / 40.0, dtype=np.float32)
                    first_day = False
                else:
                    w = bench_env.current_weights.copy()
                obs, rew, term, trunc, info = bench_env.step(w)
                done = term or trunc
                bench_equity.append(info["portfolio_value"])

            # Record Benchmark
            bench_metrics = self.calculator.compute_metrics(
                portfolio_values=bench_equity,
                trade_records=bench_env.trade_history,
                cash_history=[100000.0] + [bench_env.cash] * (len(bench_equity) - 1),
            )
            bench_metrics.update({
                "fold": f_num,
                "strategy": "Buy & Hold Benchmark",
                "seed": 42,
                "train_start": fold["train_start"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
            })
            results_records.append(bench_metrics)

            # Evaluate each strategy across multiple random seeds
            for s_name, strat, a_mode in strategies:
                for seed in self.seeds:
                    test_env = GymTradingEnv(
                        dates=fold["test_dates"],
                        action_mode=a_mode,
                        observation_mode="flattened",
                        normalizer=normalizer,
                        risk_overlay=risk_overlay,
                        initial_cash=100000.0,
                        seed=seed,
                    )
                    obs, info = test_env.reset(seed=seed)
                    done = False
                    equity_curve = [100000.0]
                    weight_hist = [np.zeros(40, dtype=np.float32)]
                    cash_hist = [100000.0]

                    while not done:
                        action = strat.predict(obs, info)
                        obs, rew, term, trunc, info = test_env.step(action)
                        done = term or trunc
                        equity_curve.append(info["portfolio_value"])
                        weight_hist.append(test_env.current_weights.copy())
                        cash_hist.append(info["cash"])

                    metrics = self.calculator.compute_metrics(
                        portfolio_values=equity_curve,
                        benchmark_values=bench_equity,
                        trade_records=test_env.trade_history,
                        cash_history=cash_hist,
                        weight_history=weight_hist,
                    )
                    metrics.update({
                        "fold": f_num,
                        "strategy": s_name,
                        "seed": seed,
                        "train_start": fold["train_start"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                    })
                    results_records.append(metrics)

        res_df = pd.DataFrame(results_records)
        res_df.to_csv(out_path / "walk_forward_per_fold_seed.csv", index=False)

        # Aggregate metrics across folds and seeds
        agg_df = (
            res_df.groupby("strategy")
            .agg(
                mean_return_pct=("total_return_pct", "mean"),
                std_return_pct=("total_return_pct", "std"),
                mean_volatility_pct=("annualized_volatility_pct", "mean"),
                mean_sharpe=("sharpe_ratio", "mean"),
                std_sharpe=("sharpe_ratio", "std"),
                mean_sortino=("sortino_ratio", "mean"),
                mean_calmar=("calmar_ratio", "mean"),
                mean_max_drawdown_pct=("max_drawdown_pct", "mean"),
                mean_var_95_pct=("historical_var_95_pct", "mean"),
                mean_cvar_95_pct=("cvar_expected_shortfall_95_pct", "mean"),
                mean_trades=("trades_count", "mean"),
                mean_fees=("total_fees_paid", "mean"),
            )
            .reset_index()
            .sort_values("mean_return_pct", ascending=False)
        )
        agg_df.to_csv(out_path / "walk_forward_aggregate_summary.csv", index=False)

        print("\n=== WALK-FORWARD AGGREGATE SUMMARY ===")
        print(agg_df.round(3).to_string(index=False))

        with open(out_path / "walk_forward_folds_manifest.json", "w") as f:
            json.dump(folds, f, indent=2, default=str)

        return agg_df
