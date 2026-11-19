from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseRewardFunction(ABC):
    """Abstract base class for environment reward calculation."""

    @abstractmethod
    def calculate(
        self,
        current_value: float,
        previous_value: float,
        current_weights: np.ndarray,
        previous_weights: np.ndarray,
        initial_capital: float = 100_000.0,
    ) -> Tuple[float, Dict[str, float]]:
        pass


class NetLogReturnReward(BaseRewardFunction):
    """Variant A: Net Log Portfolio Return.
    
    R_t = ln(V_{t+1} / V_t)
    V_{t+1} already reflects execution prices, half-spread, price impact, and transaction fees.
    """

    def calculate(
        self,
        current_value: float,
        previous_value: float,
        current_weights: np.ndarray,
        previous_weights: np.ndarray,
        initial_capital: float = 100_000.0,
    ) -> Tuple[float, Dict[str, float]]:
        ret = (current_value / max(previous_value, 1e-8)) - 1.0
        log_ret = float(np.log(max(current_value, 1e-8) / max(previous_value, 1e-8)))
        return log_ret, {
            "net_log_return": log_ret,
            "net_simple_return": ret,
        }


class TurnoverPenalizedLogReturnReward(BaseRewardFunction):
    """Variant B: Net Log Portfolio Return with Turnover Penalty.
    
    R_t = ln(V_{t+1} / V_t) - lambda_turnover * sum_i |w_{i, t+1} - w_{i, t}|
    Penalizes unnecessary churn without double-counting brokerage commissions.
    """

    def __init__(self, turnover_penalty_coef: float = 0.002):
        self.turnover_penalty_coef = float(turnover_penalty_coef)

    def calculate(
        self,
        current_value: float,
        previous_value: float,
        current_weights: np.ndarray,
        previous_weights: np.ndarray,
        initial_capital: float = 100_000.0,
    ) -> Tuple[float, Dict[str, float]]:
        log_ret = float(np.log(max(current_value, 1e-8) / max(previous_value, 1e-8)))
        turnover = float(np.sum(np.abs(current_weights - previous_weights)))
        penalty = self.turnover_penalty_coef * turnover
        reward = log_ret - penalty
        return reward, {
            "net_log_return": log_ret,
            "turnover": turnover,
            "turnover_penalty": penalty,
        }


class DownsideRiskPenalizedLogReturnReward(BaseRewardFunction):
    """Variant C: Net Log Portfolio Return with Downside-Risk (Semi-Variance) Penalty.
    
    R_t = ln(V_{t+1} / V_t) - lambda_downside * (min(0, r_t))^2
    Imposes an asymmetric quadratic penalty on negative returns.
    """

    def __init__(self, downside_penalty_coef: float = 2.0):
        self.downside_penalty_coef = float(downside_penalty_coef)

    def calculate(
        self,
        current_value: float,
        previous_value: float,
        current_weights: np.ndarray,
        previous_weights: np.ndarray,
        initial_capital: float = 100_000.0,
    ) -> Tuple[float, Dict[str, float]]:
        ret = (current_value / max(previous_value, 1e-8)) - 1.0
        log_ret = float(np.log(max(current_value, 1e-8) / max(previous_value, 1e-8)))
        downside_loss = min(0.0, ret) ** 2
        penalty = self.downside_penalty_coef * downside_loss
        reward = log_ret - penalty
        return reward, {
            "net_log_return": log_ret,
            "downside_loss": downside_loss,
            "downside_penalty": penalty,
        }


def get_reward_function(
    mode: str = "log_return",
    turnover_coef: float = 0.002,
    downside_coef: float = 2.0,
) -> BaseRewardFunction:
    """Factory creating configured reward function instance."""
    mode = str(mode).lower()
    if mode in ("log_return", "variant_a"):
        return NetLogReturnReward()
    elif mode in ("log_return_turnover_penalty", "variant_b", "turnover"):
        return TurnoverPenalizedLogReturnReward(turnover_penalty_coef=turnover_coef)
    elif mode in ("log_return_downside_penalty", "variant_c", "downside"):
        return DownsideRiskPenalizedLogReturnReward(downside_penalty_coef=downside_coef)
    else:
        return NetLogReturnReward()
