import pytest
import numpy as np

from src.agents.rule_based_agents import (
    MeanReversionAgent,
    MomentumAgent,
    VolatilityTargetingAgent,
    RiskParityAgent,
)
from src.agents.ensemble_agents import (
    EqualWeightEnsemble,
    ValidationWeightedEnsemble,
    RegimeAwareEnsemble,
)


@pytest.fixture
def constituents():
    agents = [
        VolatilityTargetingAgent(num_stocks=10),
        RiskParityAgent(num_stocks=10),
        MeanReversionAgent(num_stocks=10),
    ]
    return agents


def _mock_obs(num_stocks=10, vol=0.01, ret=0.005):
    obs = np.zeros((num_stocks, 10), dtype=np.float32)
    obs[:, 1] = ret
    obs[:, 5] = vol
    return obs.flatten()


def test_equal_weight_ensemble(constituents):
    ens = EqualWeightEnsemble(constituent_agents=constituents, num_stocks=10)
    obs = _mock_obs(10)
    w = ens.predict(obs)

    assert w.shape == (10,)
    assert np.all(w >= 0.0)
    assert np.sum(w) <= 0.95


def test_validation_weighted_ensemble_fit_and_predict(constituents):
    ens = ValidationWeightedEnsemble(constituent_agents=constituents, num_stocks=10, regularization=0.10)

    # Synthetic deterministic dev returns: Model 0 has high Sharpe, Model 1 has medium, Model 2 has negative
    T = 100
    np.random.seed(42)
    dev_returns = np.zeros((T, 3))
    dev_returns[:, 0] = 0.003 + 0.005 * np.random.randn(T)  # Sharpe ~ 9.5
    dev_returns[:, 1] = 0.001 + 0.005 * np.random.randn(T)  # Sharpe ~ 3.2
    dev_returns[:, 2] = -0.002 + 0.005 * np.random.randn(T) # Negative Sharpe

    ens.fit(dev_returns)
    assert ens.is_fitted
    assert len(ens.model_weights) == 3
    # Model 0 should receive higher weight than Model 1, Model 2 gets only shrinkage weight
    assert ens.model_weights[0] > ens.model_weights[1]
    assert ens.model_weights[1] > ens.model_weights[2]
    assert abs(np.sum(ens.model_weights) - 1.0) < 1e-5

    obs = _mock_obs(10)
    w = ens.predict(obs)
    assert w.shape == (10,)
    assert np.all(w >= 0.0)


def test_regime_aware_ensemble_regime_detection_and_logging(constituents):
    ens = RegimeAwareEnsemble(
        constituent_agents=constituents,
        num_stocks=10,
        vol_threshold=0.02,
        trend_threshold=0.01,
    )

    # 1. Stress Regime (High Vol)
    obs_stress = _mock_obs(10, vol=0.04, ret=-0.01)
    regime = ens.detect_regime(obs_stress)
    assert regime == "HIGH_VOL_STRESS"

    # 2. Bull Regime (Low Vol, High Return)
    obs_bull = _mock_obs(10, vol=0.01, ret=0.03)
    regime = ens.detect_regime(obs_bull)
    assert regime == "BULL_LOW_VOL"

    # 3. Sideways Regime
    obs_side = _mock_obs(10, vol=0.01, ret=0.002)
    regime = ens.detect_regime(obs_side)
    assert regime == "SIDEWAYS_RANGE"

    # Predict and verify audit logging
    w = ens.predict(obs_stress)
    assert len(ens.decision_history) == 1
    assert ens.decision_history[0]["regime"] == "HIGH_VOL_STRESS"


def test_continuous_rl_risk_parity_ensemble(constituents):
    from src.agents.ensemble_agents import ContinuousRLRiskParityEnsemble
    # Use 3 constituent mock agents
    ens = ContinuousRLRiskParityEnsemble(
        ppo_agent=constituents[0],
        risk_parity_agent=constituents[1],
        volatility_targeting_agent=constituents[2],
        num_stocks=10,
        weights=[0.50, 0.30, 0.20],
    )
    obs = _mock_obs(10)
    w = ens.predict(obs)

    assert ens.name == "ContinuousRLRiskParityEnsemble"
    assert w.shape == (10,)
    assert np.all(w >= 0.0)
    assert np.sum(w) <= 0.95
    assert np.allclose(ens.model_weights, [0.50, 0.30, 0.20])

