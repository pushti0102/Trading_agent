# OliveTrade 2.0 — Phase 4 Continuous RL Agents (PPO, A2C, DDPG) Report

**Date**: October 28, 2026  
**Module**: [`src/agents/continuous_rl_agents.py`](file:///d:/market-simulation/src/agents/continuous_rl_agents.py), [`src/agents/train_continuous_rl.py`](file:///d:/market-simulation/src/agents/train_continuous_rl.py)  
**Deliverables**: [`results/olive_trade_v2/phase_04_rl_agents/`](file:///d:/market-simulation/results/olive_trade_v2/phase_04_rl_agents/)  

---

## 1. Overview & Algorithm Implementations

Phase 4 implements continuous reinforcement learning agents using PyTorch and Stable-Baselines3 (`2.9.2`) interfaced with the Gymnasium trading environment (`GymTradingEnv`).

### Core Algorithms:
1. **Proximal Policy Optimization (PPO)**:
   - On-policy policy gradient method optimizing a clipped surrogate objective:
     $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
   - Architecture: Multi-layer perceptron with shared or dual heads (`[128, 128]` for policy $\pi$ and value $V$).
   - Continuous Gaussian policy producing bounded portfolio weights in $[0, 1]^{40}$.
2. **Advantage Actor-Critic (A2C)**:
   - Synchronous deterministic actor-critic with generalized advantage estimation (GAE, $\lambda=0.95$).
   - Architecture: MLP (`[128, 128]`) with learning rate $\alpha = 7 \times 10^{-4}$ and entropy regularization ($c_{\text{ent}} = 0.01$).
3. **Deep Deterministic Policy Gradient (DDPG)**:
   - Off-policy actor-critic operating over continuous action spaces.
   - Deterministic policy $\mu(s; \theta^\mu)$ paired with critic $Q(s, a; \theta^Q)$ and Polyak soft target updates ($\tau = 0.005$).
   - Exploration: Gaussian action noise $\mathcal{N}(0, 0.1)$. Replay buffer capacity: 5,000 transitions.

---

## 2. Experimental Setup & Training Protocol

- **Training Partition (70%)**: 346 sessions (`2016-01-01` to `2017-05-26`).
- **Holdout Test Partition (30%)**: 149 sessions (`2017-05-29` to `2017-12-29`).
- **State Representation**: 400-dimensional continuous vector ($40 \text{ stocks} \times 10 \text{ features}$), normalized via `ObservationNormalizer` fitted strictly on training data.
- **Action Representation**: Continuous target weight vector $\mathbf{w}_t \in [0, 1]^{40}$, projected into feasible allocations respecting the 5% cash floor and 20% max concentration cap.
- **Reward Function**: Log portfolio return with turnover penalty:
  $$R_t = \ln\left(\frac{V_{t+1}}{V_t}\right) - \lambda_{\text{turnover}} \sum_{i=1}^N |w_{i, t+1} - w_{i, t}|$$

---

## 3. Holdout Evaluation Results (149 Trading Days)

All agents evaluated on unseen holdout dates starting from **₹100,000.00**:

| Algorithm | Action Mode | Initial Equity (₹) | Final Equity (₹) | Total Return % | Ann. Volatility % | Sharpe Ratio | Max Drawdown % | Trades Count | Total Fees (₹) |
|---|---|---|---|---|---|---|---|---|---|
| **A2C** | Continuous | 100,000.00 | 88,841.35 | **-11.16%** | 11.46% | -1.70 | -14.57% | 1,444 | ₹1,231.88 |
| **PPO** | Continuous | 100,000.00 | 76,733.14 | **-23.27%** | 11.69% | -3.80 | -26.14% | 2,125 | ₹2,783.85 |
| **DDPG** | Continuous | 100,000.00 | 63,928.92 | **-36.07%** | 12.59% | -5.98 | -36.55% | 2,083 | ₹2,922.30 |

---

## 4. Empirical Microstructure Diagnostics

1. **Rebalancing Friction Penalty**:
   Continuous RL policies operating on daily rebalancing intervals generated between 1,400 and 2,125 trades over 149 days ($\sim 10$ to $14$ trades per day). With a 2% quoted bid-ask spread and quadratic impact, transaction costs consumed ₹1,230 to ₹2,920 in explicit fees, alongside substantial execution slippage.
2. **Actor-Critic vs Off-Policy DDPG**:
   A2C demonstrated superior sample stability under financial friction compared to DDPG, limiting total drawdown to -14.57% compared to DDPG's -36.55%.
3. **Action Distribution & Concentration**:
   All three algorithms maintained diversified multi-asset portfolios across 23 to 33 active constituent positions, adhering to single-stock concentration caps.
4. **Transition to Phase 5**:
   These findings confirm the necessity of Phase 5 (turnover damping overlays, volatility targeting, and downside risk penalties) to transition continuous RL policies into profitable net-of-fee regimes.

---

## 5. Phase 4 Deliverables

Archived in [`results/olive_trade_v2/phase_04_rl_agents/`](file:///d:/market-simulation/results/olive_trade_v2/phase_04_rl_agents/):
- `checkpoints/ppo_model.zip`: Serialized PPO policy checkpoint.
- `checkpoints/a2c_model.zip`: Serialized A2C policy checkpoint.
- `checkpoints/ddpg_model.zip`: Serialized DDPG policy checkpoint.
- `continuous_rl_comparison.csv`: Performance metrics across all algorithms.
- `action_exposure_stats.json`: Portfolio exposure, position counts, and cash allocations.
- `training_summary.json`: Training runtime, timesteps, and convergence metrics.
- `rl_hyperparameters.json`: Complete hyperparameter configuration manifest.
- `phase_04_report.md`: This comprehensive evaluation report.

Phase 4 acceptance criteria are fulfilled.
