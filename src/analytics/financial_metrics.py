from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd


class FinancialMetricsCalculator:
    """Comprehensive Institutional Financial Risk and Performance Analytics Engine.

    Computes:
    - Return metrics: Total Return, Annualized Return (CAGR), Benchmark-Relative Return.
    - Volatility & Risk-adjusted ratios: Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Information Ratio.
    - Tail Risk: Maximum Drawdown, Drawdown Duration, Historical VaR (95%, 99%), Historical CVaR/Expected Shortfall (95%, 99%).
    - Execution & Microstructure: Turnover, Number of Trades, Average Holding Period, Spread Cost, Impact Cost, Fees.
    - Portfolio Composition & Compliance: Average Cash %, Herfindahl Concentration (HHI), Risk-Limit Violations.
    """

    def __init__(self, risk_free_rate: float = 0.05, trading_days_per_year: int = 252):
        self.risk_free_rate = float(risk_free_rate)
        self.trading_days = int(trading_days_per_year)
        self.daily_rf = self.risk_free_rate / self.trading_days

    def compute_metrics(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        benchmark_values: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        trade_records: Optional[List[Dict[str, Any]]] = None,
        cash_history: Optional[List[float]] = None,
        weight_history: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute complete institutional risk metrics.

        Handles edge cases: zero volatility, zero drawdown, negative CAGR gracefully without NaNs.
        """
        vals = np.asarray(portfolio_values, dtype=np.float64)
        if len(vals) < 2:
            return self._empty_metrics(initial_capital=float(vals[0]) if len(vals) > 0 else 100_000.0)

        initial_val = float(vals[0])
        final_val = float(vals[-1])
        n_days = len(vals) - 1

        # Daily returns
        daily_returns = np.diff(vals) / np.maximum(vals[:-1], 1e-8)
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Total Return & Annualized Return (CAGR)
        total_return = (final_val / max(initial_val, 1e-8)) - 1.0
        if final_val > 0 and initial_val > 0 and n_days > 0:
            ann_return = float((final_val / initial_val) ** (self.trading_days / n_days) - 1.0)
        else:
            ann_return = -1.0 if final_val <= 0 else 0.0

        # 2. Volatility
        daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
        ann_volatility = daily_std * np.sqrt(self.trading_days)

        # 3. Sharpe Ratio
        excess_daily_mean = float(np.mean(daily_returns) - self.daily_rf)
        if daily_std > 1e-8:
            sharpe_ratio = float((excess_daily_mean / daily_std) * np.sqrt(self.trading_days))
        else:
            sharpe_ratio = 0.0

        # 4. Sortino Ratio (Downside Semi-Deviation)
        downside_diffs = np.minimum(0.0, daily_returns - self.daily_rf)
        downside_variance = float(np.mean(downside_diffs ** 2))
        downside_std = np.sqrt(downside_variance)
        if daily_std > 1e-8 and downside_std > 1e-8:
            sortino_ratio = float((excess_daily_mean / downside_std) * np.sqrt(self.trading_days))
        else:
            sortino_ratio = 0.0

        # 5. Drawdown & Drawdown Duration
        peaks = np.maximum.accumulate(vals)
        drawdowns = (vals - peaks) / np.maximum(peaks, 1e-8)
        max_drawdown = float(np.min(drawdowns))  # negative float

        # Calculate max drawdown duration (consecutive days below previous high-water mark)
        current_dd_duration = 0
        max_dd_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        # 6. Calmar Ratio
        if abs(max_drawdown) > 1e-6:
            calmar_ratio = float(ann_return / abs(max_drawdown))
        else:
            calmar_ratio = 0.0

        # 7. Historical VaR & CVaR (Expected Shortfall) at 95% and 99%
        # Sign convention: reported as positive percentage loss (e.g., 2.5% loss)
        sorted_returns = np.sort(daily_returns)
        var_95_idx = int(np.floor(0.05 * len(sorted_returns)))
        var_99_idx = int(np.floor(0.01 * len(sorted_returns)))

        var_95 = float(-sorted_returns[min(var_95_idx, len(sorted_returns) - 1)])
        var_99 = float(-sorted_returns[min(var_99_idx, len(sorted_returns) - 1)])

        tail_95 = sorted_returns[: max(1, var_95_idx + 1)]
        cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else var_95

        tail_99 = sorted_returns[: max(1, var_99_idx + 1)]
        cvar_99 = float(-np.mean(tail_99)) if len(tail_99) > 0 else var_99

        # 8. Benchmark-relative analytics
        benchmark_return = None
        information_ratio = None
        tracking_error = None
        if benchmark_values is not None:
            b_vals = np.asarray(benchmark_values, dtype=np.float64)
            if len(b_vals) == len(vals):
                b_init = float(b_vals[0])
                b_final = float(b_vals[-1])
                benchmark_return = float((b_final / max(b_init, 1e-8)) - 1.0)

                b_returns = np.diff(b_vals) / np.maximum(b_vals[:-1], 1e-8)
                diff_returns = daily_returns - b_returns
                te_daily = float(np.std(diff_returns, ddof=1)) if len(diff_returns) > 1 else 0.0
                tracking_error = float(te_daily * np.sqrt(self.trading_days))

                if tracking_error > 1e-8:
                    information_ratio = float((np.mean(diff_returns) * self.trading_days) / tracking_error)
                else:
                    information_ratio = 0.0

        # 9. Microstructure & Trade Statistics
        trades = trade_records or []
        num_trades = len(trades)
        total_fees = float(sum(t.get("fee", 0.0) for t in trades))
        
        # Estimate spread cost & market impact cost
        spread_costs = 0.0
        impact_costs = 0.0
        for t in trades:
            tv = float(t.get("trade_value", 0.0))
            half_spread = float(t.get("spread_rate", 0.02)) / 2.0
            impact = float(t.get("impact_rate", 0.0))
            spread_costs += tv * half_spread
            impact_costs += tv * impact

        # 10. Portfolio Weights, Turnover, and Concentration
        turnover = 0.0
        avg_cash_pct = 100.0
        hhi_concentration = 0.0
        risk_violations = 0

        if weight_history and len(weight_history) > 1:
            w_arr = np.array(weight_history)
            turnover = float(np.mean(np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)))
            # Herfindahl index (sum of squared stock weights)
            hhi_concentration = float(np.mean(np.sum(w_arr ** 2, axis=1)))
            # Check for weight violations > 20%
            risk_violations += int(np.sum(w_arr > 0.2001))

        if cash_history and len(cash_history) == len(vals):
            cash_fractions = np.array(cash_history) / np.maximum(vals, 1e-8)
            avg_cash_pct = float(np.mean(cash_fractions) * 100.0)
            # Check for cash floor violations < 4.99%
            risk_violations += int(np.sum(cash_fractions < 0.0499))

        avg_holding_period = float(1.0 / max(turnover, 1e-4)) if turnover > 0 else float(n_days)

        return {
            "initial_equity": initial_val,
            "final_equity": final_val,
            "total_return_pct": float(total_return * 100.0),
            "annualized_return_pct": float(ann_return * 100.0),
            "annualized_volatility_pct": float(ann_volatility * 100.0),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown_pct": float(max_drawdown * 100.0),
            "max_drawdown_duration_days": int(max_dd_duration),
            "historical_var_95_pct": float(var_95 * 100.0),
            "historical_var_99_pct": float(var_99 * 100.0),
            "cvar_expected_shortfall_95_pct": float(cvar_95 * 100.0),
            "cvar_expected_shortfall_99_pct": float(cvar_99 * 100.0),
            "benchmark_return_pct": float(benchmark_return * 100.0) if benchmark_return is not None else None,
            "tracking_error_pct": float(tracking_error * 100.0) if tracking_error is not None else None,
            "information_ratio": float(information_ratio) if information_ratio is not None else None,
            "trades_count": int(num_trades),
            "total_fees_paid": float(total_fees),
            "estimated_spread_costs": float(spread_costs),
            "estimated_impact_costs": float(impact_costs),
            "average_turnover": float(turnover),
            "average_holding_period_days": float(avg_holding_period),
            "average_cash_allocation_pct": float(avg_cash_pct),
            "hhi_concentration": float(hhi_concentration),
            "risk_limit_violations": int(risk_violations),
        }

    def _empty_metrics(self, initial_capital: float) -> Dict[str, Any]:
        return {
            "initial_equity": initial_capital,
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration_days": 0,
            "historical_var_95_pct": 0.0,
            "historical_var_99_pct": 0.0,
            "cvar_expected_shortfall_95_pct": 0.0,
            "cvar_expected_shortfall_99_pct": 0.0,
            "benchmark_return_pct": None,
            "tracking_error_pct": None,
            "information_ratio": None,
            "trades_count": 0,
            "total_fees_paid": 0.0,
            "estimated_spread_costs": 0.0,
            "estimated_impact_costs": 0.0,
            "average_turnover": 0.0,
            "average_holding_period_days": 0.0,
            "average_cash_allocation_pct": 100.0,
            "hhi_concentration": 0.0,
            "risk_limit_violations": 0,
        }
