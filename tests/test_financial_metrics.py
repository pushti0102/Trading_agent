import pytest
import numpy as np

from src.analytics.financial_metrics import FinancialMetricsCalculator


def test_returns_and_cagr():
    calc = FinancialMetricsCalculator(risk_free_rate=0.0, trading_days_per_year=252)

    # 252 trading days, steady growth from 100k to 110k (+10%)
    vals = np.linspace(100_000.0, 110_000.0, 253)
    metrics = calc.compute_metrics(vals)

    assert abs(metrics["total_return_pct"] - 10.0) < 0.1
    assert abs(metrics["annualized_return_pct"] - 10.0) < 0.2
    assert metrics["max_drawdown_pct"] == 0.0
    assert metrics["max_drawdown_duration_days"] == 0


def test_sharpe_and_sortino_ratios():
    calc = FinancialMetricsCalculator(risk_free_rate=0.04, trading_days_per_year=252)

    # Returns with known mean and variance
    np.random.seed(42)
    daily_rets = np.random.normal(0.001, 0.01, size=250)
    vals = [100_000.0]
    for r in daily_rets:
        vals.append(vals[-1] * (1.0 + r))

    metrics = calc.compute_metrics(vals)
    assert metrics["annualized_volatility_pct"] > 0.0
    assert np.isfinite(metrics["sharpe_ratio"])
    assert np.isfinite(metrics["sortino_ratio"])


def test_drawdown_and_duration():
    calc = FinancialMetricsCalculator()

    # Equity: 100k -> 120k -> 96k (20% drawdown) -> 120k
    vals = [100_000.0, 110_000.0, 120_000.0, 108_000.0, 96_000.0, 108_000.0, 120_000.0]
    metrics = calc.compute_metrics(vals)

    # Max DD should be (96k - 120k) / 120k = -20.0%
    assert abs(metrics["max_drawdown_pct"] - (-20.0)) < 1e-4
    # Spent 3 days below peak (108k, 96k, 108k)
    assert metrics["max_drawdown_duration_days"] >= 3
    assert metrics["calmar_ratio"] != 0.0


def test_var_and_cvar_expected_shortfall():
    calc = FinancialMetricsCalculator()

    # Normal returns with known std = 0.02
    np.random.seed(123)
    daily_rets = np.random.normal(0.0, 0.02, size=500)
    vals = [100_000.0]
    for r in daily_rets:
        vals.append(vals[-1] * (1.0 + r))

    metrics = calc.compute_metrics(vals)

    # VaR95 should be approximately 1.645 * 2% = ~3.3%
    assert 2.0 < metrics["historical_var_95_pct"] < 5.0
    # VaR99 > VaR95
    assert metrics["historical_var_99_pct"] >= metrics["historical_var_95_pct"]
    # CVaR (expected shortfall) > VaR
    assert metrics["cvar_expected_shortfall_95_pct"] >= metrics["historical_var_95_pct"]
    assert metrics["cvar_expected_shortfall_99_pct"] >= metrics["historical_var_99_pct"]


def test_benchmark_tracking_error_and_information_ratio():
    calc = FinancialMetricsCalculator()

    vals = np.linspace(100_000.0, 110_000.0, 100)
    b_vals = np.linspace(100_000.0, 105_000.0, 100)

    metrics = calc.compute_metrics(vals, benchmark_values=b_vals)
    assert metrics["benchmark_return_pct"] is not None
    assert abs(metrics["benchmark_return_pct"] - 5.0) < 0.1
    assert metrics["tracking_error_pct"] is not None
    assert metrics["information_ratio"] is not None


def test_zero_volatility_and_edge_cases():
    calc = FinancialMetricsCalculator()

    # Flat equity line
    vals = [100_000.0] * 50
    metrics = calc.compute_metrics(vals)

    assert metrics["annualized_volatility_pct"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0
    assert np.isfinite(metrics["historical_var_95_pct"])
