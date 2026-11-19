import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.environment.gym_trading_env import GymTradingEnv, ObservationNormalizer
from src.risk.risk_overlay import PortfolioRiskOverlay
from src.agents.continuous_rl_agents import PPOAgentWrapper, A2CAgentWrapper, DDPGAgentWrapper
from src.agents.rule_based_agents import (
    MeanReversionAgent,
    VolatilityTargetingAgent,
    RiskParityAgent,
    StackelbergBestResponseAgent,
)
from src.agents.ensemble_agents import (
    EqualWeightEnsemble,
    ValidationWeightedEnsemble,
    RegimeAwareEnsemble,
)


def run_phase_5_and_6_experiments():
    base_results = Path("results/olive_trade_v2")
    p5_dir = base_results / "phase_05_risk_management"
    p6_dir = base_results / "phase_05_ensembles"
    p5_dir.mkdir(parents=True, exist_ok=True)
    p6_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dates
    df = pd.read_csv("data/processed/selected_stocks.csv", parse_dates=["TIMESTAMP"])
    all_dates = sorted(df["TIMESTAMP"].unique())
    split_70 = int(len(all_dates) * 0.70)
    train_dates = all_dates[:split_70]
    test_dates = all_dates[split_70:]

    # Further split 70% train into dev for ensemble fitting (zero leakage from test!)
    dev_split = int(len(train_dates) * 0.70)
    pure_train_dates = train_dates[:dev_split]
    dev_dates = train_dates[dev_split:]

    # Load frozen normalizer
    norm_path = base_results / "phase_02_environment" / "normalizer_params.json"
    with open(norm_path, "r") as f:
        norm_dict = json.load(f)
    normalizer = ObservationNormalizer.from_dict(norm_dict)

    # 2. Load trained continuous RL models from Phase 4
    p4_checkpoints = base_results / "phase_04_rl_agents" / "checkpoints"
    ppo = PPOAgentWrapper.load(str(p4_checkpoints / "ppo_model.zip"))
    a2c = A2CAgentWrapper.load(str(p4_checkpoints / "a2c_model.zip"))
    ddpg = DDPGAgentWrapper.load(str(p4_checkpoints / "ddpg_model.zip"))

    vol_target = VolatilityTargetingAgent(num_stocks=40, target_annual_vol=0.12)
    risk_parity = RiskParityAgent(num_stocks=40)
    mean_rev = MeanReversionAgent(num_stocks=40)
    stack_br = StackelbergBestResponseAgent(num_stocks=40)

    # Helper evaluation function
    def evaluate_policy(agent, env, initial_cash=100000.0):
        obs, info = env.reset(seed=42)
        done = False
        daily_equity = [initial_cash]
        daily_returns = []

        while not done:
            action = agent.predict(obs, info)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            daily_equity.append(info["portfolio_value"])
            daily_returns.append(info.get("step_return", 0.0))

        vals = np.array(daily_equity)
        returns = np.diff(vals) / vals[:-1]
        tot_ret = (vals[-1] / vals[0] - 1.0) * 100.0
        daily_std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        ann_vol = daily_std * np.sqrt(252) * 100.0
        sharpe = (np.mean(returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
        peaks = np.maximum.accumulate(vals)
        dd = (vals - peaks) / peaks
        max_dd = np.min(dd) * 100.0

        return {
            "initial_equity": float(vals[0]),
            "final_equity": float(vals[-1]),
            "total_return_pct": float(tot_ret),
            "ann_volatility_pct": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd),
            "trades_count": int(env.total_trades_count),
            "fees_paid": float(env.total_fees_paid),
            "daily_returns": daily_returns,
        }

    # ==========================================
    # EXPERIMENT 1: Phase 5 Risk & Reward Overlay
    # ==========================================
    print("\n--- Running Phase 5 Reward & Risk Overlay Experiments ---")
    reward_modes = [
        ("Variant A: Net Log Return", "log_return", 0.001),
        ("Variant B: Turnover Penalty", "log_return_turnover_penalty", 0.002),
        ("Variant C: Downside Penalty", "log_return_downside_penalty", 0.001),
    ]

    p5_reward_results = []
    for label, mode, t_coef in reward_modes:
        env = GymTradingEnv(
            dates=test_dates,
            action_mode="continuous",
            observation_mode="flattened",
            normalizer=normalizer,
            reward_mode=mode,
            turnover_penalty_coef=t_coef,
            initial_cash=100000.0,
            seed=42,
        )
        res = evaluate_policy(vol_target, env)
        res_summary = {k: v for k, v in res.items() if k != "daily_returns"}
        res_summary["reward_variant"] = label
        p5_reward_results.append(res_summary)

    p5_rew_df = pd.DataFrame(p5_reward_results)
    p5_rew_df.to_csv(p5_dir / "reward_variants_comparison.csv", index=False)

    # Risk Overlay Ablations
    risk_configs = [
        ("Full Risk Overlay", PortfolioRiskOverlay(max_single_stock_weight=0.20, cash_reserve_fraction=0.05, max_turnover_per_step=0.20, volatility_target=0.12, max_drawdown_limit=0.10)),
        ("No Turnover Damping", PortfolioRiskOverlay(max_single_stock_weight=0.20, cash_reserve_fraction=0.05, max_turnover_per_step=None, volatility_target=0.12, max_drawdown_limit=0.10)),
        ("No Volatility Target", PortfolioRiskOverlay(max_single_stock_weight=0.20, cash_reserve_fraction=0.05, max_turnover_per_step=0.20, volatility_target=None, max_drawdown_limit=0.10)),
        ("No Circuit Breaker", PortfolioRiskOverlay(max_single_stock_weight=0.20, cash_reserve_fraction=0.05, max_turnover_per_step=0.20, volatility_target=0.12, max_drawdown_limit=None)),
    ]

    p5_risk_results = []
    for label, overlay in risk_configs:
        env = GymTradingEnv(
            dates=test_dates,
            action_mode="continuous",
            observation_mode="flattened",
            normalizer=normalizer,
            risk_overlay=overlay,
            initial_cash=100000.0,
            seed=42,
        )
        res = evaluate_policy(vol_target, env)
        res_summary = {k: v for k, v in res.items() if k != "daily_returns"}
        res_summary["risk_configuration"] = label
        p5_risk_results.append(res_summary)

    p5_risk_df = pd.DataFrame(p5_risk_results)
    p5_risk_df.to_csv(p5_dir / "risk_overlay_ablation.csv", index=False)
    print("Saved Phase 5 results to:", p5_dir)

    # ==========================================
    # EXPERIMENT 2: Phase 6 Ensemble Strategies
    # ==========================================
    print("\n--- Running Phase 6 Ensemble Strategies ---")

    # Constituents list for multi-asset continuous allocation
    constituents = [
        vol_target,
        risk_parity,
        mean_rev,
        ppo,
        a2c,
    ]

    # 1. Fit ValidationWeightedEnsemble on Development Interval (Zero Test Leakage)
    print("Fitting Validation-Weighted Ensemble on development interval (106 dates)...")
    dev_returns = []
    for agent in constituents:
        dev_env = GymTradingEnv(
            dates=dev_dates,
            action_mode="continuous",
            observation_mode="flattened",
            normalizer=normalizer,
            initial_cash=100000.0,
            seed=42,
        )
        res = evaluate_policy(agent, dev_env)
        dev_returns.append(res["daily_returns"])

    dev_matrix = np.column_stack(dev_returns)
    val_ensemble = ValidationWeightedEnsemble(
        constituent_agents=constituents,
        num_stocks=40,
        regularization=0.20,
    )
    val_ensemble.fit(dev_matrix)

    # 2. Equal Weight Ensemble
    eq_ensemble = EqualWeightEnsemble(constituent_agents=constituents, num_stocks=40)

    # 3. Regime Aware Ensemble
    regime_ensemble = RegimeAwareEnsemble(
        constituent_agents=constituents,
        num_stocks=40,
        vol_threshold=0.015,
        trend_threshold=0.005,
    )

    # 4. Benchmark & Model Evaluation on Holdout Test Partition (149 dates)
    all_models = [
        ("Buy & Hold Baseline", "benchmark"),
        ("PPO Agent", ppo),
        ("A2C Agent", a2c),
        ("DDPG Agent", ddpg),
        ("Volatility Targeting", vol_target),
        ("Risk Parity", risk_parity),
        ("Mean Reversion", mean_rev),
        ("Equal-Weight Ensemble", eq_ensemble),
        ("Validation-Weighted Ensemble", val_ensemble),
        ("Regime-Aware Ensemble", regime_ensemble),
    ]

    p6_results = []
    # Test environment with full risk overlay
    test_risk_overlay = PortfolioRiskOverlay(
        max_single_stock_weight=0.20,
        cash_reserve_fraction=0.05,
        max_turnover_per_step=0.25,
        volatility_target=0.12,
        max_drawdown_limit=0.10,
    )

    for name, model in all_models:
        env = GymTradingEnv(
            dates=test_dates,
            action_mode="continuous",
            observation_mode="flattened",
            normalizer=normalizer,
            risk_overlay=test_risk_overlay,
            initial_cash=100000.0,
            seed=42,
        )

        if name == "Buy & Hold Baseline":
            # Buy & hold day-1 entry equal weight across 40 stocks
            obs, info = env.reset(seed=42)
            done = False
            daily_equity = [100000.0]
            first_day = True
            while not done:
                if first_day:
                    w = np.full(40, 0.90 / 40.0, dtype=np.float32)
                    first_day = False
                else:
                    w = env.current_weights.copy()
                obs, rew, term, trunc, info = env.step(w)
                done = term or trunc
                daily_equity.append(info["portfolio_value"])

            vals = np.array(daily_equity)
            returns = np.diff(vals) / vals[:-1]
            tot_ret = (vals[-1] / vals[0] - 1.0) * 100.0
            daily_std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
            ann_vol = daily_std * np.sqrt(252) * 100.0
            sharpe = (np.mean(returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            peaks = np.maximum.accumulate(vals)
            dd = (vals - peaks) / peaks
            max_dd = np.min(dd) * 100.0

            res = {
                "initial_equity": float(vals[0]),
                "final_equity": float(vals[-1]),
                "total_return_pct": float(tot_ret),
                "ann_volatility_pct": float(ann_vol),
                "sharpe_ratio": float(sharpe),
                "max_drawdown_pct": float(max_dd),
                "trades_count": int(env.total_trades_count),
                "fees_paid": float(env.total_fees_paid),
            }
        else:
            raw_res = evaluate_policy(model, env)
            res = {k: v for k, v in raw_res.items() if k != "daily_returns"}

        res["model"] = name
        p6_results.append(res)

    p6_df = pd.DataFrame(p6_results).sort_values("total_return_pct", ascending=False)
    p6_df.to_csv(p6_dir / "ensemble_comparison.csv", index=False)
    print("\n=== PHASE 6 ENSEMBLE STRATEGY COMPARISON ===")
    print(p6_df.round(3).to_string(index=False))

    # Save ensemble metadata
    ens_meta = {
        "validation_weights": {agent.name: float(w) for agent, w in zip(constituents, val_ensemble.model_weights)},
        "dev_dates_count": len(dev_dates),
        "test_dates_count": len(test_dates),
        "regime_distribution": pd.Series([d["regime"] for d in regime_ensemble.decision_history]).value_counts().to_dict(),
    }
    with open(p6_dir / "ensemble_metadata.json", "w") as f:
        json.dump(ens_meta, f, indent=2)

    return p5_rew_df, p5_risk_df, p6_df


if __name__ == "__main__":
    run_phase_5_and_6_experiments()
