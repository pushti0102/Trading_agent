from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.rl_trader import STATE_SIZE


class MeanReversionAgent(BaseAgent):
    """Mean-Reversion Strategy.
    
    Extracts valuation spread (feature 0: price / fundamental - 1.0).
    Allocates positive weights to undervalued stocks (price < fundamental)
    and zero / underweight to overvalued stocks.
    """

    def __init__(self, num_stocks: int = 40, threshold: float = -0.01, max_weight: float = 0.15):
        super().__init__(name="MeanReversion", config={"threshold": threshold, "max_weight": max_weight})
        self.num_stocks = num_stocks
        self.threshold = threshold
        self.max_weight = max_weight

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        val_spreads = obs[:, 0]  # feature 0: price / fundamental - 1.0

        weights = np.zeros(self.num_stocks, dtype=np.float32)
        # Identify undervalued stocks
        undervalued_mask = val_spreads < self.threshold
        if np.any(undervalued_mask):
            raw_scores = -val_spreads[undervalued_mask]  # more negative spread = higher score
            score_sum = np.sum(raw_scores)
            if score_sum > 1e-8:
                allocated = 0.90 * (raw_scores / score_sum)
                weights[undervalued_mask] = np.clip(allocated, 0.0, self.max_weight)

        total = np.sum(weights)
        if total > 0.95:
            weights = (weights / total) * 0.95
        return weights


class MomentumAgent(BaseAgent):
    """Cross-Sectional Momentum Strategy.
    
    Extracts daily return (feature 1) and assigns higher target weights to top performers.
    """

    def __init__(self, num_stocks: int = 40, top_k: int = 10, max_weight: float = 0.15):
        super().__init__(name="Momentum", config={"top_k": top_k, "max_weight": max_weight})
        self.num_stocks = num_stocks
        self.top_k = min(top_k, num_stocks)
        self.max_weight = max_weight

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        returns = obs[:, 1]  # feature 1: daily return

        weights = np.zeros(self.num_stocks, dtype=np.float32)
        top_indices = np.argsort(returns)[-self.top_k:]

        # Filter to strictly positive momentum
        valid_indices = [idx for idx in top_indices if returns[idx] > 0.0]
        if valid_indices:
            per_stock_weight = min(0.90 / len(valid_indices), self.max_weight)
            weights[valid_indices] = per_stock_weight

        total = np.sum(weights)
        if total > 0.95:
            weights = (weights / total) * 0.95
        return weights


class TrendFilteredMomentumAgent(BaseAgent):
    """Trend-Filtered Momentum Strategy.
    
    Combines momentum ranking with a moving-average regime filter.
    Momentum longs are only initiated if the price is above its 20-day anchor (feature 0 > 0),
    hedging against bear-market momentum crashes.
    """

    def __init__(self, num_stocks: int = 40, top_k: int = 8, max_weight: float = 0.15):
        super().__init__(name="TrendFilteredMomentum", config={"top_k": top_k, "max_weight": max_weight})
        self.num_stocks = num_stocks
        self.top_k = top_k
        self.max_weight = max_weight

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        val_spreads = obs[:, 0]  # price vs fundamental anchor
        returns = obs[:, 1]

        # Stock must have positive momentum AND be trading above fundamental anchor (uptrend)
        uptrend_mask = (val_spreads >= 0.0) & (returns > 0.0)
        weights = np.zeros(self.num_stocks, dtype=np.float32)

        candidate_indices = np.where(uptrend_mask)[0]
        if len(candidate_indices) > 0:
            candidate_returns = returns[candidate_indices]
            top_ranked = candidate_indices[np.argsort(candidate_returns)[-self.top_k:]]
            equal_w = min(0.90 / len(top_ranked), self.max_weight)
            weights[top_ranked] = equal_w

        return weights


class VolatilityTargetingAgent(BaseAgent):
    """Volatility-Targeting Strategy.
    
    Scales portfolio risk based on 20-day rolling volatility (feature 5).
    Weights are inversely proportional to volatility, targeting a stable annualized volatility.
    """

    def __init__(self, num_stocks: int = 40, target_annual_vol: float = 0.12, max_weight: float = 0.15):
        super().__init__(name="VolatilityTargeting", config={"target_annual_vol": target_annual_vol, "max_weight": max_weight})
        self.num_stocks = num_stocks
        self.target_annual_vol = target_annual_vol
        self.target_daily_vol = target_annual_vol / np.sqrt(252)
        self.max_weight = max_weight

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        vols = np.clip(obs[:, 5], 0.005, 0.20)  # feature 5: rolling return std

        inv_vol = 1.0 / vols
        weights = inv_vol / np.sum(inv_vol)

        # Scale total portfolio exposure based on average market volatility
        avg_vol = np.mean(vols)
        leverage_factor = np.clip(self.target_daily_vol / max(avg_vol, 1e-4), 0.2, 0.95)

        scaled_weights = np.clip(weights * leverage_factor, 0.0, self.max_weight)
        total = np.sum(scaled_weights)
        if total > 0.95:
            scaled_weights = (scaled_weights / total) * 0.95
        return scaled_weights.astype(np.float32)


class RiskParityAgent(BaseAgent):
    """Equal Risk Contribution (Risk Parity) Strategy.
    
    Allocates portfolio weights such that each asset contributes equally to total risk,
    approximated under diagonal covariance: w_i = (1 / sigma_i) / sum(1 / sigma_j).
    """

    def __init__(self, num_stocks: int = 40, max_weight: float = 0.15):
        super().__init__(name="RiskParity", config={"max_weight": max_weight})
        self.num_stocks = num_stocks
        self.max_weight = max_weight

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        vols = np.clip(obs[:, 5], 0.005, 0.20)

        inv_vols = 1.0 / vols
        raw_w = inv_vols / np.sum(inv_vols)

        # Cap individual weights and renormalize
        capped_w = np.clip(raw_w * 0.90, 0.0, self.max_weight)
        total = np.sum(capped_w)
        if total > 0.95:
            capped_w = (capped_w / total) * 0.95
        return capped_w.astype(np.float32)


class StackelbergLeaderAgent(BaseAgent):
    """Original First-Mover Stackelberg Leader Policy.
    
    Buys 5 shares when price < fundamental * 0.99, sells 5 shares when price > fundamental * 1.01.
    Mapped to discrete actions {0: BUY, 1: HOLD, 2: SELL} per stock.
    """

    def __init__(self, num_stocks: int = 40):
        super().__init__(name="StackelbergLeader")
        self.num_stocks = num_stocks

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        val_spreads = obs[:, 0]  # price / fundamental - 1.0

        actions = np.ones(self.num_stocks, dtype=np.int64)  # default HOLD
        actions[val_spreads < -0.01] = 0  # BUY
        actions[val_spreads > 0.01] = 2   # SELL
        return actions


class StackelbergFollowerAgent(BaseAgent):
    """Original Second-Mover Reactive Follower Policy.
    
    Responds to leader orders and 1-day price momentum.
    """

    def __init__(self, num_stocks: int = 40):
        super().__init__(name="StackelbergFollower")
        self.num_stocks = num_stocks

    def predict(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        leader_actions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        returns = obs[:, 1]

        actions = np.ones(self.num_stocks, dtype=np.int64)

        for i in range(self.num_stocks):
            l_act = leader_actions[i] if leader_actions is not None else 1
            if l_act == 0 and returns[i] >= 0:
                actions[i] = 0  # BUY
            elif l_act == 2:
                actions[i] = 2  # SELL
            elif returns[i] > 0.01:
                actions[i] = 0  # BUY
            elif returns[i] < -0.01:
                actions[i] = 2  # SELL
            else:
                actions[i] = 1  # HOLD

        return actions


class StackelbergBestResponseAgent(BaseAgent):
    """Stackelberg Best-Response Equilibrium Solver.
    
    Leader explicitly models the follower's best-response reaction function:
        a_F^*(a_L) = argmax_{a_F in A_F} U_F(a_L, a_F)
    And solves for the optimal commitment:
        a_L^* = argmax_{a_L in A_L} U_L(a_L, a_F^*(a_L))
    
    Evaluates expected net payoff accounting for follower price impact.
    """

    def __init__(self, num_stocks: int = 40, impact_coef: float = 0.0005, spread: float = 0.02):
        super().__init__(name="StackelbergBestResponse", config={"impact_coef": impact_coef, "spread": spread})
        self.num_stocks = num_stocks
        self.impact_coef = impact_coef
        self.half_spread = spread / 2.0

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(self.num_stocks, STATE_SIZE)
        val_spreads = obs[:, 0]  # price divergence
        returns = obs[:, 1]
        vols = obs[:, 5]

        leader_actions = np.ones(self.num_stocks, dtype=np.int64)

        for i in range(self.num_stocks):
            spread_div = val_spreads[i]
            ret = returns[i]
            vol = max(vols[i], 0.005)

            # Follower reaction function:
            # If Leader BUY: Follower buys if ret >= 0, adding positive price impact
            # If Leader SELL: Follower sells, depressing exit price
            
            # Candidate Leader actions: {0: BUY, 1: HOLD, 2: SELL}
            # Expected net utility for Leader taking a_L given follower best response:
            payoff_buy = 0.0
            payoff_hold = 0.0
            payoff_sell = 0.0

            # 1. Action = BUY
            # Anticipated follower reaction:
            follower_reaction_to_buy = 0 if ret >= 0 else 1
            follower_impact = self.impact_coef * 5 if follower_reaction_to_buy == 0 else 0.0
            # Expected reversion gain: -spread_div, minus execution friction, plus follower demand push
            friction = self.half_spread + self.impact_coef * 5
            payoff_buy = (-spread_div) + follower_impact - friction

            # 2. Action = HOLD
            payoff_hold = 0.0

            # 3. Action = SELL
            # Follower follows sell
            friction_sell = self.half_spread + self.impact_coef * 5
            payoff_sell = spread_div - friction_sell

            # Leader optimal commitment
            payoffs = [payoff_buy, payoff_hold, payoff_sell]
            best_action = int(np.argmax(payoffs))

            # Only trade if expected profit exceeds a risk margin proportional to volatility
            risk_hurdle = 0.5 * vol
            if best_action == 0 and payoff_buy > risk_hurdle:
                leader_actions[i] = 0
            elif best_action == 2 and payoff_sell > risk_hurdle:
                leader_actions[i] = 2
            else:
                leader_actions[i] = 1

        return leader_actions
