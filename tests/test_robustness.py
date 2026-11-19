import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluation.robustness import RobustnessEvaluator
from src.risk.risk_overlay import PortfolioRiskOverlay


@pytest.fixture(scope="module")
def evaluator():
    return RobustnessEvaluator(default_seed=42)


def test_evaluator_initialization(evaluator):
    """Verify train/holdout split and normalizer initialization."""
    assert len(evaluator.all_dates) == 495
    assert len(evaluator.train_dates) == 346
    assert len(evaluator.holdout_dates) == 149
    assert evaluator.normalizer.fitted is True
    assert evaluator.normalizer.mean is not None


def test_cost_sensitivity_monotonic_friction(evaluator):
    """Verify that wider spreads strictly increase spread cost and friction INR on benchmark."""
    test_dates = evaluator.holdout_dates[:30]

    sim_tight = evaluator.run_strategy_simulation(
        strategy_name="Buy & Hold Benchmark",
        dates=test_dates,
        spread_rate=0.002,  # 0.2% quoted (0.1% half-spread)
        impact_coeff=0.0005,
        transaction_cost=0.001,
        seed=42,
    )

    sim_wide = evaluator.run_strategy_simulation(
        strategy_name="Buy & Hold Benchmark",
        dates=test_dates,
        spread_rate=0.030,  # 3.0% quoted (1.5% half-spread)
        impact_coeff=0.0005,
        transaction_cost=0.001,
        seed=42,
    )

    spread_cost_tight = sim_tight["metrics"]["spread_cost_inr"]
    spread_cost_wide = sim_wide["metrics"]["spread_cost_inr"]
    friction_tight = sim_tight["metrics"]["total_friction_inr"]
    friction_wide = sim_wide["metrics"]["total_friction_inr"]

    assert spread_cost_wide > spread_cost_tight * 10  # 1.5% vs 0.1% is 15x
    assert friction_wide > friction_tight
    assert sim_wide["metrics"]["quoted_spread_pct"] == 3.0
    assert sim_wide["metrics"]["half_spread_pct"] == 1.5


def test_market_regime_stress_execution(evaluator):
    """Verify strategy simulation across custom regime date slice."""
    regime_dates = evaluator.holdout_dates[:40]

    sim = evaluator.run_strategy_simulation(
        strategy_name="Volatility Targeting",
        dates=regime_dates,
        spread_rate=0.02,
        impact_coeff=0.0005,
        seed=42,
    )

    m = sim["metrics"]
    assert "total_return_pct" in m
    assert "annualized_volatility_pct" in m
    assert "sharpe_ratio" in m
    assert "max_drawdown_pct" in m
    assert len(sim["equity_curve"]) == len(regime_dates)


def test_block_bootstrap_confidence_interval_bounds(evaluator):
    """Verify moving block bootstrap produces consistent, bounded confidence intervals."""
    # Run with small B and small strategy set for speed
    boot_df = evaluator.compute_block_bootstrap_confidence_intervals(
        strategies=["Risk Parity"],
        block_size=5,
        n_bootstraps=50,
        seed=42,
    )

    assert len(boot_df) == 1
    row = boot_df.iloc[0]
    assert row["boot_return_ci_lower_pct"] <= row["boot_return_ci_upper_pct"]
    assert row["boot_vol_ci_lower_pct"] <= row["boot_vol_ci_upper_pct"]
    assert row["boot_sharpe_ci_lower"] <= row["boot_sharpe_ci_upper"]
    assert row["boot_max_dd_ci_lower_pct"] <= row["boot_max_dd_ci_upper_pct"]


def test_risk_overlay_ablation_turnover_damping(evaluator):
    """Verify institutional turnover damping reduces cumulative friction."""
    test_dates = evaluator.holdout_dates[:50]

    overlay_unconstrained = None
    overlay_damped = PortfolioRiskOverlay(
        cash_reserve_fraction=0.05,
        max_single_stock_weight=0.20,
        max_turnover_per_step=0.15,
    )

    sim_unconstrained = evaluator.run_strategy_simulation(
        strategy_name="Mean Reversion",
        dates=test_dates,
        risk_overlay=overlay_unconstrained,
        seed=42,
    )

    sim_damped = evaluator.run_strategy_simulation(
        strategy_name="Mean Reversion",
        dates=test_dates,
        risk_overlay=overlay_damped,
        seed=42,
    )

    # Damping reduces total friction
    f_unconstrained = sim_unconstrained["metrics"]["total_friction_inr"]
    f_damped = sim_damped["metrics"]["total_friction_inr"]
    assert f_damped < f_unconstrained
