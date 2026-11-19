import pytest
import numpy as np
import pandas as pd

from src.environment.gym_trading_env import GymTradingEnv, ObservationNormalizer


def test_gym_env_spaces_discrete():
    env = GymTradingEnv(
        symbols=["RELIANCE"],
        action_mode="discrete",
        observation_mode="single_stock",
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (10,)
    assert np.isfinite(obs).all()
    assert env.action_space.n == 3

    next_obs, reward, terminated, truncated, info = env.step(0)  # BUY
    assert next_obs.shape == (10,)
    assert np.isfinite(reward)
    assert not truncated


def test_gym_env_spaces_continuous():
    symbols = ["RELIANCE", "INFY", "TCS", "HDFC"]
    env = GymTradingEnv(
        symbols=symbols,
        action_mode="continuous",
        observation_mode="flattened",
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (len(symbols) * 10,)
    assert np.isfinite(obs).all()
    assert env.action_space.shape == (len(symbols),)

    target_weights = np.array([0.15, 0.15, 0.10, 0.10], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(target_weights)

    assert next_obs.shape == (len(symbols) * 10,)
    assert np.isfinite(reward)
    assert info["portfolio_value"] > 0


def test_cash_conservation_and_inventory():
    symbols = ["RELIANCE", "INFY"]
    env = GymTradingEnv(
        symbols=symbols,
        action_mode="continuous",
        initial_cash=100_000.0,
        cash_reserve_fraction=0.05,
    )
    obs, info = env.reset(seed=42)
    initial_cash = env.cash

    # Target: 20% in RELIANCE, 0% in INFY
    target_weights = np.array([0.20, 0.0], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(target_weights)

    # Cash must have decreased by trade value + fee
    assert env.cash < initial_cash
    assert env.inventory["RELIANCE"] > 0
    assert env.inventory["INFY"] == 0

    # Valuation check
    expected_val = env.cash + env.inventory["RELIANCE"] * env.last_prices["RELIANCE"]
    assert abs(env.portfolio_value - expected_val) < 1e-4


def test_cash_reserve_floor_enforced():
    symbols = ["RELIANCE", "INFY", "TCS"]
    env = GymTradingEnv(
        symbols=symbols,
        action_mode="continuous",
        initial_cash=100_000.0,
        cash_reserve_fraction=0.10,  # 10% reserve = ₹10,000 floor
    )
    env.reset(seed=42)

    # Try to allocate 100% across stocks
    extreme_weights = np.array([0.33, 0.33, 0.33], dtype=np.float32)
    env.step(extreme_weights)

    # Cash must never drop below the reserve floor
    min_reserve = 100_000.0 * 0.10
    assert env.cash >= min_reserve - 1e-2


def test_position_concentration_cap_enforced():
    symbols = ["RELIANCE", "INFY"]
    env = GymTradingEnv(
        symbols=symbols,
        action_mode="continuous",
        initial_cash=100_000.0,
        max_position_fraction=0.20,  # 20% max cap
    )
    env.reset(seed=42)

    # Request 80% in RELIANCE
    excessive_weights = np.array([0.80, 0.0], dtype=np.float32)
    projected = env._project_target_weights(excessive_weights)

    assert projected[0] <= 0.20 + 1e-6


def test_observation_normalizer_strictly_fitted_on_train():
    # Fit normalizer on synthetic training data
    train_data = np.random.normal(loc=10.0, scale=2.0, size=(100, 10)).astype(np.float32)
    normalizer = ObservationNormalizer(clip_min=-5.0, clip_max=5.0)
    normalizer.fit(train_data)

    assert normalizer.fitted
    assert abs(normalizer.mean.mean() - 10.0) < 0.5

    # Test observation transformation
    test_obs = np.array([10.0] * 10, dtype=np.float32)
    norm_obs = normalizer.transform(test_obs)
    # 10.0 should normalize to approx 0.0
    assert np.allclose(norm_obs, 0.0, atol=0.5)

    # Serialization test
    d = normalizer.to_dict()
    restored = ObservationNormalizer.from_dict(d)
    assert restored.fitted
    assert np.allclose(restored.mean, normalizer.mean)


def test_chronological_advancement_and_termination():
    # Use small subset of dates
    env = GymTradingEnv(symbols=["RELIANCE"])
    all_dates = list(env.feed.dates)[:5]
    env._restrict_feed_dates(all_dates)
    env.reset(seed=42)

    steps = 0
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(np.array([0.1], dtype=np.float32))
        steps += 1
        done = term or trunc
        if steps > 10:
            break

    # Must terminate exactly when dates are exhausted
    assert done
    assert steps == len(all_dates) - 1
