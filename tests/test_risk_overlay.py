import pytest
import numpy as np

from src.risk.risk_overlay import PortfolioRiskOverlay
from src.environment.reward_functions import (
    NetLogReturnReward,
    TurnoverPenalizedLogReturnReward,
    DownsideRiskPenalizedLogReturnReward,
    get_reward_function,
)


def test_reward_function_variants():
    # 1. Variant A: Net Log Return
    rew_a = NetLogReturnReward()
    r, info = rew_a.calculate(
        current_value=105_000.0,
        previous_value=100_000.0,
        current_weights=np.zeros(10),
        previous_weights=np.zeros(10),
    )
    assert abs(r - np.log(1.05)) < 1e-6
    assert abs(info["net_simple_return"] - 0.05) < 1e-6

    # 2. Variant B: Turnover penalty
    rew_b = TurnoverPenalizedLogReturnReward(turnover_penalty_coef=0.01)
    w1 = np.array([0.1, 0.2])
    w0 = np.array([0.0, 0.0])  # turnover = 0.3
    r, info = rew_b.calculate(
        current_value=100_000.0,
        previous_value=100_000.0,
        current_weights=w1,
        previous_weights=w0,
    )
    expected_penalty = 0.01 * 0.3
    assert abs(r - (-expected_penalty)) < 1e-6

    # 3. Variant C: Downside risk penalty
    rew_c = DownsideRiskPenalizedLogReturnReward(downside_penalty_coef=2.0)
    # Negative return of -5%
    r, info = rew_c.calculate(
        current_value=95_000.0,
        previous_value=100_000.0,
        current_weights=np.zeros(2),
        previous_weights=np.zeros(2),
    )
    expected_downside = (-0.05) ** 2
    expected_r = np.log(0.95) - 2.0 * expected_downside
    assert abs(r - expected_r) < 1e-5


def test_risk_overlay_cash_reserve_and_caps():
    overlay = PortfolioRiskOverlay(
        max_single_stock_weight=0.15,
        cash_reserve_fraction=0.10,  # Max equity = 90%
        max_turnover_per_step=None,
    )

    # Raw weights requesting 30% on each of 4 stocks (total 120%)
    raw_weights = np.array([0.30, 0.30, 0.30, 0.30], dtype=np.float32)
    feasible_w, diag = overlay.apply(raw_weights)

    assert diag["cash_reserve_enforced"]
    assert np.sum(feasible_w) <= 0.90 + 1e-6
    assert np.all(feasible_w <= 0.15 + 1e-6)
    assert np.all(feasible_w >= 0.0)


def test_risk_overlay_turnover_damping():
    overlay = PortfolioRiskOverlay(
        max_single_stock_weight=0.20,
        cash_reserve_fraction=0.05,
        max_turnover_per_step=0.20,
    )

    current_w = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    target_w = np.array([0.20, 0.20, 0.20, 0.20], dtype=np.float32)

    damped_w, diag = overlay.apply(target_w, current_weights=current_w)

    assert diag["turnover_damping_applied"]
    # Total turnover must be constrained
    step_turnover = np.sum(np.abs(damped_w - current_w))
    assert step_turnover < np.sum(np.abs(target_w - current_w))


def test_risk_overlay_circuit_breaker_drawdown():
    overlay = PortfolioRiskOverlay(
        max_single_stock_weight=0.20,
        cash_reserve_fraction=0.05,
        max_drawdown_limit=0.10,  # 10% DD limit
        drawdown_deleveraging_factor=0.50,
        max_turnover_per_step=None,
    )

    target_w = np.array([0.15, 0.15, 0.15], dtype=np.float32)

    # Normal state (-5% DD) -> No circuit breaker
    w_normal, diag_normal = overlay.apply(target_w, current_drawdown=-0.05)
    assert not diag_normal["circuit_breaker_triggered"]
    assert np.allclose(w_normal, target_w)

    # Severe state (-15% DD) -> Circuit breaker cuts equity in half
    w_deleveraged, diag_severe = overlay.apply(target_w, current_drawdown=-0.15)
    assert diag_severe["circuit_breaker_triggered"]
    assert np.allclose(w_deleveraged, target_w * 0.50)


def test_risk_overlay_volatility_targeting():
    overlay = PortfolioRiskOverlay(
        max_single_stock_weight=0.20,
        cash_reserve_fraction=0.05,
        volatility_target=0.10,  # 10% target vol
        max_turnover_per_step=None,
    )

    target_w = np.array([0.10, 0.10, 0.10], dtype=np.float32)

    # High market volatility (20%) -> Exposure scaled down by 0.10 / 0.20 = 0.5x
    w_scaled, diag = overlay.apply(target_w, market_volatility=0.20)
    assert diag["volatility_scaling_applied"]
    assert np.allclose(w_scaled, target_w * 0.50)


def test_risk_overlay_delta_w_max_and_deadband():
    """Verify |w_{i,t} - w_{i,t-1}| <= delta_w_max and micro-trade deadband suppression."""
    overlay = PortfolioRiskOverlay(
        max_single_stock_weight=0.20,
        cash_reserve_fraction=0.05,
        max_turnover_per_step=None,
        delta_w_max=0.04,  # Cap delta at 4%
        rebalance_deadband=0.015,  # 1.5% deadband threshold
    )

    current_w = np.array([0.05, 0.10, 0.08, 0.00], dtype=np.float32)
    # Asset 0: shift from 0.05 to 0.058 (+0.008, < 0.015 deadband) -> should be kept at 0.05
    # Asset 1: shift from 0.10 to 0.20 (+0.10, > delta_w_max 0.04) -> should be capped at 0.14
    # Asset 2: shift from 0.08 to 0.02 (-0.06, > delta_w_max 0.04) -> should be capped at 0.04
    # Asset 3: shift from 0.00 to 0.03 (+0.03, > deadband and <= delta_w_max) -> should become 0.03
    target_w = np.array([0.058, 0.20, 0.02, 0.03], dtype=np.float32)

    feasible_w, diag = overlay.apply(target_w, current_weights=current_w)

    assert diag["turnover_damping_applied"]
    # Asset 0 shift suppressed by deadband
    assert abs(feasible_w[0] - 0.05) < 1e-4
    # Asset 1 shift clamped by delta_w_max (0.10 + 0.04 = 0.14)
    assert abs(feasible_w[1] - 0.14) < 1e-4
    # Asset 2 shift clamped by delta_w_max (0.08 - 0.04 = 0.04)
    assert abs(feasible_w[2] - 0.04) < 1e-4
    # Asset 3 executed
    assert abs(feasible_w[3] - 0.03) < 1e-4

