import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.environment.gym_trading_env import GymTradingEnv
from src.agents.continuous_rl_agents import (
    create_ppo_agent,
    create_a2c_agent,
    create_ddpg_agent,
    PPOAgentWrapper,
    A2CAgentWrapper,
    DDPGAgentWrapper,
)


@pytest.fixture
def dummy_env():
    # 4 stocks for fast unit testing
    symbols = ["RELIANCE", "INFY", "TCS", "HDFC"]
    env = GymTradingEnv(
        symbols=symbols,
        action_mode="continuous",
        observation_mode="flattened",
    )
    return env


def test_create_and_predict_ppo(dummy_env):
    agent = create_ppo_agent(dummy_env, n_steps=16, batch_size=8, seed=42)
    obs, _ = dummy_env.reset(seed=42)
    action = agent.predict(obs)

    assert action.shape == (dummy_env.num_stocks,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)


def test_create_and_predict_a2c(dummy_env):
    agent = create_a2c_agent(dummy_env, n_steps=8, seed=42)
    obs, _ = dummy_env.reset(seed=42)
    action = agent.predict(obs)

    assert action.shape == (dummy_env.num_stocks,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)


def test_create_and_predict_ddpg(dummy_env):
    agent = create_ddpg_agent(dummy_env, buffer_size=500, batch_size=16, seed=42)
    obs, _ = dummy_env.reset(seed=42)
    action = agent.predict(obs)

    assert action.shape == (dummy_env.num_stocks,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)


def test_smoke_train_and_save_ppo(dummy_env):
    agent = create_ppo_agent(dummy_env, n_steps=16, batch_size=8, seed=42)
    agent.model.learn(total_timesteps=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ppo_test.zip"
        agent.save(str(save_path))
        assert save_path.exists()

        loaded = PPOAgentWrapper.load(str(save_path), env=dummy_env)
        obs, _ = dummy_env.reset(seed=42)
        act = loaded.predict(obs)
        assert act.shape == (dummy_env.num_stocks,)


def test_smoke_train_and_save_a2c(dummy_env):
    agent = create_a2c_agent(dummy_env, n_steps=8, seed=42)
    agent.model.learn(total_timesteps=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "a2c_test.zip"
        agent.save(str(save_path))
        assert save_path.exists()


def test_smoke_train_and_save_ddpg(dummy_env):
    agent = create_ddpg_agent(dummy_env, buffer_size=500, batch_size=16, seed=42)
    agent.model.learn(total_timesteps=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ddpg_test.zip"
        agent.save(str(save_path))
        assert save_path.exists()
