import pytest
import numpy as np
import torch

from src.agents.rule_based_agents import (
    MeanReversionAgent,
    MomentumAgent,
    TrendFilteredMomentumAgent,
    VolatilityTargetingAgent,
    RiskParityAgent,
    StackelbergLeaderAgent,
    StackelbergFollowerAgent,
    StackelbergBestResponseAgent,
)
from src.agents.double_dueling_dqn import (
    SumTree,
    PrioritizedReplayBuffer,
    DuelingQNetwork,
    DoubleDuelingDQNAgent,
)


def _create_mock_observation(num_stocks=40):
    # (num_stocks, 10)
    obs = np.zeros((num_stocks, 10), dtype=np.float32)
    # stock 0: undervalued (-5%), positive return (+1%), low vol (0.01)
    obs[0, 0] = -0.05
    obs[0, 1] = 0.01
    obs[0, 5] = 0.01

    # stock 1: overvalued (+5%), negative return (-2%), high vol (0.05)
    obs[1, 0] = 0.05
    obs[1, 1] = -0.02
    obs[1, 5] = 0.05

    # remaining stocks: neutral
    obs[2:, 0] = 0.0
    obs[2:, 1] = 0.005
    obs[2:, 5] = 0.02
    return obs.flatten()


def test_mean_reversion_agent():
    agent = MeanReversionAgent(num_stocks=40, threshold=-0.01)
    obs = _create_mock_observation(40)
    weights = agent.predict(obs)

    assert weights.shape == (40,)
    assert weights[0] > 0.0  # Undervalued stock receives long allocation
    assert weights[1] == 0.0  # Overvalued stock receives 0
    assert np.sum(weights) <= 0.95


def test_momentum_agent():
    agent = MomentumAgent(num_stocks=40, top_k=5)
    obs = _create_mock_observation(40)
    weights = agent.predict(obs)

    assert weights.shape == (40,)
    assert weights[0] > 0.0  # stock 0 has positive return
    assert weights[1] == 0.0  # stock 1 has negative return


def test_trend_filtered_momentum_agent():
    agent = TrendFilteredMomentumAgent(num_stocks=40, top_k=5)
    obs = _create_mock_observation(40)
    weights = agent.predict(obs)

    assert weights.shape == (40,)
    # stock 0 has positive return but is below fundamental anchor (spread = -0.05 < 0)
    # so trend filter screens it out
    assert weights[0] == 0.0
    assert np.all(weights >= 0.0)


def test_volatility_targeting_agent():
    agent = VolatilityTargetingAgent(num_stocks=40, target_annual_vol=0.12)
    obs = _create_mock_observation(40)
    weights = agent.predict(obs)

    assert weights.shape == (40,)
    assert np.all(weights >= 0.0)
    # Lower vol stock 0 should receive higher weight than higher vol stock 1
    assert weights[0] > weights[1]
    assert np.sum(weights) <= 0.95


def test_risk_parity_agent():
    agent = RiskParityAgent(num_stocks=40)
    obs = _create_mock_observation(40)
    weights = agent.predict(obs)

    assert weights.shape == (40,)
    assert np.all(weights >= 0.0)
    assert weights[0] > weights[1]
    assert np.sum(weights) <= 0.95


def test_stackelberg_best_response_agent():
    agent = StackelbergBestResponseAgent(num_stocks=40)
    obs = _create_mock_observation(40)
    actions = agent.predict(obs)

    assert actions.shape == (40,)
    # Actions must be in {0, 1, 2}
    assert np.all(np.isin(actions, [0, 1, 2]))


def test_sum_tree_and_prioritized_buffer():
    buffer = PrioritizedReplayBuffer(capacity=100)
    s = np.zeros(10, dtype=np.float32)
    next_s = np.zeros(10, dtype=np.float32)

    for i in range(20):
        buffer.add(s, 0, 1.0, next_s, False)

    assert len(buffer) == 20
    batch, indices, is_weights = buffer.sample(batch_size=8)
    assert len(batch[0]) == 8
    assert len(indices) == 8
    assert len(is_weights) == 8
    assert np.all(is_weights > 0.0)

    # Priority update
    td_errors = np.array([0.5] * 8)
    buffer.update_priorities(indices, td_errors)
    assert buffer.tree.total() > 0


def test_dueling_q_network_forward():
    net = DuelingQNetwork(state_size=10, action_size=3, hidden_dim=64)
    x = torch.randn(4, 10)
    q_vals = net(x)

    assert q_vals.shape == (4, 3)
    assert torch.isfinite(q_vals).all()


def test_double_dueling_dqn_agent_learning_step():
    agent = DoubleDuelingDQNAgent(
        state_size=10,
        action_size=3,
        batch_size=8,
        memory_size=100,
        learning_rate=0.001,
        seed=42,
    )

    s = np.random.randn(10).astype(np.float32)
    act = agent.act(s, training=True)
    assert act in (0, 1, 2)

    # Populate replay buffer
    for _ in range(16):
        agent.remember(s, 0, 1.0, s, False)

    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0
    assert agent.training_steps == 1
