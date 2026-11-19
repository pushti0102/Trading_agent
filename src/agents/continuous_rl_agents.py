from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.base_agent import BaseAgent


class TrainingMetricsCallback(BaseCallback):
    """Logs episode reward, portfolio equity, and step returns during SB3 training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.step_records = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            self.step_records.append({
                "step": self.num_timesteps,
                "portfolio_value": info.get("portfolio_value", 0.0),
                "total_trades": info.get("total_trades", 0),
                "fees_paid": info.get("total_fees_paid", 0.0),
                "turnover": info.get("turnover", 0.0),
            })
        return True


class PPOAgentWrapper(BaseAgent):
    """Wrapper around Stable-Baselines3 PPO for continuous portfolio-weight allocation."""

    def __init__(self, model: PPO, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PPO", config=config or {})
        self.model = model

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None) -> "PPOAgentWrapper":
        model = PPO.load(path, env=env)
        return cls(model=model)


class A2CAgentWrapper(BaseAgent):
    """Wrapper around Stable-Baselines3 A2C for continuous portfolio-weight allocation."""

    def __init__(self, model: A2C, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="A2C", config=config or {})
        self.model = model

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None) -> "A2CAgentWrapper":
        model = A2C.load(path, env=env)
        return cls(model=model)


class DDPGAgentWrapper(BaseAgent):
    """Wrapper around Stable-Baselines3 DDPG for continuous portfolio-weight allocation."""

    def __init__(self, model: DDPG, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="DDPG", config=config or {})
        self.model = model

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None) -> "DDPGAgentWrapper":
        model = DDPG.load(path, env=env)
        return cls(model=model)


def create_ppo_agent(
    env: gym.Env,
    learning_rate: float = 3e-4,
    n_steps: int = 64,
    batch_size: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    seed: int = 42,
    hidden_dim: int = 128,
) -> PPOAgentWrapper:
    """Factory creating a tuned continuous PPO agent."""
    policy_kwargs = dict(
        net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim]),
        activation_fn=torch.nn.ReLU,
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )
    config = {
        "algorithm": "PPO",
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "seed": seed,
    }
    return PPOAgentWrapper(model=model, config=config)


def create_a2c_agent(
    env: gym.Env,
    learning_rate: float = 7e-4,
    n_steps: int = 16,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    seed: int = 42,
    hidden_dim: int = 128,
) -> A2CAgentWrapper:
    """Factory creating a tuned continuous A2C agent."""
    policy_kwargs = dict(
        net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim]),
        activation_fn=torch.nn.ReLU,
    )
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )
    config = {
        "algorithm": "A2C",
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gamma": gamma,
        "seed": seed,
    }
    return A2CAgentWrapper(model=model, config=config)


def create_ddpg_agent(
    env: gym.Env,
    learning_rate: float = 1e-3,
    buffer_size: int = 10000,
    batch_size: int = 64,
    gamma: float = 0.99,
    tau: float = 0.005,
    noise_sigma: float = 0.1,
    seed: int = 42,
    hidden_dim: int = 128,
) -> DDPGAgentWrapper:
    """Factory creating a tuned continuous DDPG agent with Gaussian exploration noise."""
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_sigma * np.ones(n_actions),
    )
    policy_kwargs = dict(
        net_arch=dict(pi=[hidden_dim, hidden_dim], qf=[hidden_dim, hidden_dim]),
        activation_fn=torch.nn.ReLU,
    )
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )
    config = {
        "algorithm": "DDPG",
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "noise_sigma": noise_sigma,
        "seed": seed,
    }
    return DDPGAgentWrapper(model=model, config=config)
