from typing import Dict, Any, Optional, Tuple
import numpy as np


class PortfolioRiskOverlay:
    """Institutional Portfolio Risk Overlay and Feasible-Action Projector.

    Enforces:
    1. Maximum single-stock allocation cap (default 20%).
    2. Cash reserve floor (default 5%).
    3. Maximum turnover per rebalancing step (damping rebalancing churn).
    4. Optional dynamic volatility targeting.
    5. Optional maximum drawdown intervention (circuit breaker).
    6. Non-negative long-only portfolio constraint.
    """

    def __init__(
        self,
        max_single_stock_weight: float = 0.20,
        cash_reserve_fraction: float = 0.05,
        max_order_size: int = 100,
        max_turnover_per_step: Optional[float] = 0.25,
        delta_w_max: Optional[float] = None,
        rebalance_deadband: Optional[float] = None,
        volatility_target: Optional[float] = None,
        max_drawdown_limit: Optional[float] = 0.10,
        drawdown_deleveraging_factor: float = 0.50,
    ):
        self.max_single_stock_weight = float(max_single_stock_weight)
        self.cash_reserve_fraction = float(cash_reserve_fraction)
        self.max_order_size = int(max_order_size)
        self.max_turnover_per_step = (
            float(max_turnover_per_step) if max_turnover_per_step is not None else None
        )
        self.delta_w_max = float(delta_w_max) if delta_w_max is not None else None
        self.rebalance_deadband = float(rebalance_deadband) if rebalance_deadband is not None else None
        self.volatility_target = (
            float(volatility_target) if volatility_target is not None else None
        )
        self.max_drawdown_limit = (
            float(max_drawdown_limit) if max_drawdown_limit is not None else None
        )
        self.drawdown_deleveraging_factor = float(drawdown_deleveraging_factor)

    def apply(
        self,
        target_weights: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        portfolio_value: float = 100_000.0,
        current_drawdown: float = 0.0,
        market_volatility: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Project raw target weights into feasible risk-compliant weights.

        Args:
            target_weights: Raw continuous output in [0, 1]^N.
            current_weights: Currently realized weights (N,).
            portfolio_value: Current mark-to-market equity.
            current_drawdown: Peak-to-trough drawdown (e.g. -0.08 for -8%).
            market_volatility: Current annualized or rolling volatility.

        Returns:
            feasible_weights: Bounded, constraint-satisfying weight vector.
            diagnostics: Audit metadata on triggered risk constraints.
        """
        w = np.clip(np.asarray(target_weights, dtype=np.float64), 0.0, None)
        num_stocks = len(w)
        if current_weights is None:
            current_weights = np.zeros(num_stocks, dtype=np.float64)
        else:
            current_weights = np.asarray(current_weights, dtype=np.float64)

        diagnostics = {
            "circuit_breaker_triggered": False,
            "volatility_scaling_applied": False,
            "turnover_damping_applied": False,
            "concentration_cap_applied": False,
            "cash_reserve_enforced": False,
            "raw_total_weight": float(np.sum(w)),
        }

        # 1. Drawdown Circuit Breaker Intervention
        # If drawdown breaches limit (e.g. DD < -10%), scale down total equity allocation
        if self.max_drawdown_limit is not None and current_drawdown < -self.max_drawdown_limit:
            w = w * self.drawdown_deleveraging_factor
            diagnostics["circuit_breaker_triggered"] = True

        # 2. Volatility Targeting Scaling
        if self.volatility_target is not None and market_volatility is not None:
            if market_volatility > 1e-4:
                vol_scale = min(1.0, self.volatility_target / market_volatility)
                if vol_scale < 1.0:
                    w = w * vol_scale
                    diagnostics["volatility_scaling_applied"] = True

        # 3. Cash Reserve Floor constraint
        max_investable = max(0.0, 1.0 - self.cash_reserve_fraction)
        total_w = np.sum(w)
        if total_w > max_investable:
            w = (w / total_w) * max_investable
            diagnostics["cash_reserve_enforced"] = True

        # 4. Maximum Single-Stock Concentration Cap
        over_cap_mask = w > self.max_single_stock_weight
        if np.any(over_cap_mask):
            w[over_cap_mask] = self.max_single_stock_weight
            diagnostics["concentration_cap_applied"] = True

        # 5. Turnover Damping Constraint: |w_{i, t} - w_{i, t-1}| <= \Delta w_{\max}
        # Prevents high-frequency micro-trade churning and clips whipsaws
        if self.delta_w_max is not None or self.max_turnover_per_step is not None:
            delta = w - current_weights

            # Optional rebalance deadband: suppress micro-trades below threshold
            if self.rebalance_deadband is not None and self.rebalance_deadband > 0.0:
                under_threshold = np.abs(delta) < self.rebalance_deadband
                delta[under_threshold] = 0.0

            if self.delta_w_max is not None:
                max_delta = self.delta_w_max
            else:
                max_delta = self.max_turnover_per_step / num_stocks * 2.0

            clipped_delta = np.clip(delta, -max_delta, max_delta)
            if not np.allclose(w - current_weights, clipped_delta):
                w = current_weights + clipped_delta
                # Re-verify boundaries
                w = np.clip(w, 0.0, self.max_single_stock_weight)
                total_w = np.sum(w)
                if total_w > max_investable:
                    w = (w / total_w) * max_investable
                diagnostics["turnover_damping_applied"] = True

        # Final validation
        w = np.clip(w, 0.0, self.max_single_stock_weight)
        final_total = np.sum(w)
        if final_total > max_investable:
            w = (w / final_total) * max_investable

        diagnostics["final_total_weight"] = float(np.sum(w))
        diagnostics["final_cash_fraction"] = float(1.0 - np.sum(w))

        return w.astype(np.float32), diagnostics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_single_stock_weight": self.max_single_stock_weight,
            "cash_reserve_fraction": self.cash_reserve_fraction,
            "max_order_size": self.max_order_size,
            "max_turnover_per_step": self.max_turnover_per_step,
            "delta_w_max": self.delta_w_max,
            "rebalance_deadband": self.rebalance_deadband,
            "volatility_target": self.volatility_target,
            "max_drawdown_limit": self.max_drawdown_limit,
            "drawdown_deleveraging_factor": self.drawdown_deleveraging_factor,
        }
