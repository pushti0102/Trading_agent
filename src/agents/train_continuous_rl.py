import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.environment.gym_trading_env import GymTradingEnv, ObservationNormalizer
from src.agents.continuous_rl_agents import (
    create_ppo_agent,
    create_a2c_agent,
    create_ddpg_agent,
    TrainingMetricsCallback,
)


def train_and_evaluate_continuous_rl(
    train_timesteps: int = 1000,
    out_dir_path: str = "results/olive_trade_v2/phase_04_rl_agents",
    seed: int = 42,
):
    out_dir = Path(out_dir_path)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dates
    df = pd.read_csv("data/processed/selected_stocks.csv", parse_dates=["TIMESTAMP"])
    all_dates = sorted(df["TIMESTAMP"].unique())
    split = int(len(all_dates) * 0.70)
    train_dates = all_dates[:split]
    test_dates = all_dates[split:]

    # 2. Fit normalizer on training data
    print("Fitting ObservationNormalizer on 70% train split...")
    temp_train_env = GymTradingEnv(
        dates=train_dates,
        action_mode="continuous",
        observation_mode="flattened",
    )
    obs_samples = []
    obs, _ = temp_train_env.reset(seed=seed)
    obs_samples.append(obs)
    done = False
    while not done:
        act = np.random.uniform(0.0, 0.05, size=temp_train_env.num_stocks).astype(np.float32)
        obs, rew, term, trunc, _ = temp_train_env.step(act)
        obs_samples.append(obs)
        done = term or trunc

    normalizer = ObservationNormalizer(clip_min=-5.0, clip_max=5.0)
    normalizer.fit(np.array(obs_samples))

    # 3. Training environments with turnover penalty reward
    def make_train_env():
        return GymTradingEnv(
            dates=train_dates,
            action_mode="continuous",
            observation_mode="flattened",
            reward_mode="log_return_turnover_penalty",
            turnover_penalty_coef=0.002,
            normalizer=normalizer,
            seed=seed,
        )

    # 4. Instantiate agents
    print(f"Initializing PPO, A2C, and DDPG agents (training {train_timesteps} steps each)...")
    ppo_env = make_train_env()
    ppo_agent = create_ppo_agent(ppo_env, learning_rate=3e-4, n_steps=64, batch_size=32, seed=seed)

    a2c_env = make_train_env()
    a2c_agent = create_a2c_agent(a2c_env, learning_rate=7e-4, n_steps=16, seed=seed)

    ddpg_env = make_train_env()
    ddpg_agent = create_ddpg_agent(ddpg_env, learning_rate=1e-3, buffer_size=5000, batch_size=64, seed=seed)

    # 5. Train agents
    models = [
        ("PPO", ppo_agent, ppo_env),
        ("A2C", a2c_agent, a2c_env),
        ("DDPG", ddpg_agent, ddpg_env),
    ]

    training_logs = {}
    for name, agent, env in models:
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        cb = TrainingMetricsCallback()
        agent.model.learn(total_timesteps=train_timesteps, callback=cb)
        elapsed = time.time() - t0
        save_path = checkpoints_dir / f"{name.lower()}_model.zip"
        agent.save(str(save_path))
        print(f"Saved {name} checkpoint: {save_path} ({elapsed:.1f}s)")
        training_logs[name] = {
            "training_time_sec": elapsed,
            "timesteps": train_timesteps,
            "final_train_portfolio_value": env.portfolio_value,
            "train_trades": env.total_trades_count,
            "train_fees": env.total_fees_paid,
        }

    # 6. Evaluate on Holdout Partition
    print("\nEvaluating trained PPO, A2C, and DDPG on Holdout Test Partition (149 days)...")
    eval_results = []
    action_stats = {}

    for name, agent, _ in models:
        test_env = GymTradingEnv(
            dates=test_dates,
            action_mode="continuous",
            observation_mode="flattened",
            normalizer=normalizer,
            initial_cash=100000.0,
            seed=seed,
        )
        obs, info = test_env.reset(seed=seed)
        done = False
        daily_equity = [100000.0]
        recorded_weights = []

        while not done:
            action = agent.predict(obs, info)
            recorded_weights.append(action)
            obs, rew, term, trunc, info = test_env.step(action)
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

        all_w = np.array(recorded_weights)
        mean_weights = np.mean(all_w, axis=0)
        total_exposure = np.sum(all_w, axis=1)

        action_stats[name] = {
            "mean_total_exposure": float(np.mean(total_exposure)),
            "max_single_stock_weight": float(np.max(all_w)),
            "mean_cash_fraction": float(1.0 - np.mean(total_exposure)),
            "active_positions_count": int(np.sum(mean_weights > 0.01)),
        }

        eval_results.append({
            "algorithm": name,
            "action_mode": "continuous",
            "initial_equity": float(vals[0]),
            "final_equity": float(vals[-1]),
            "total_return_pct": float(tot_ret),
            "ann_volatility_pct": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd),
            "trades_count": int(test_env.total_trades_count),
            "fees_paid": float(test_env.total_fees_paid),
        })

    eval_df = pd.DataFrame(eval_results).sort_values("total_return_pct", ascending=False)
    eval_df.to_csv(out_dir / "continuous_rl_comparison.csv", index=False)
    print("\n=== PHASE 4 CONTINUOUS RL EVALUATION ===")
    print(eval_df.round(3).to_string(index=False))

    with open(out_dir / "action_exposure_stats.json", "w") as f:
        json.dump(action_stats, f, indent=2)

    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(training_logs, f, indent=2)

    # Hyperparameter configs
    hp_config = {
        "PPO": ppo_agent.get_config(),
        "A2C": a2c_agent.get_config(),
        "DDPG": ddpg_agent.get_config(),
        "reward_mode": "log_return_turnover_penalty",
        "turnover_penalty_coef": 0.002,
        "cash_reserve_floor": 0.05,
        "max_position_fraction": 0.20,
    }
    with open(out_dir / "rl_hyperparameters.json", "w") as f:
        json.dump(hp_config, f, indent=2)

    return eval_df, action_stats


if __name__ == "__main__":
    train_and_evaluate_continuous_rl(train_timesteps=1500)
