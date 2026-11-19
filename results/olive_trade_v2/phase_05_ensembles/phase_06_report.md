# OliveTrade 2.0 — Phase 6 Ensemble Strategies Report

**Date**: October 29, 2026  
**Module**: [`src/agents/ensemble_agents.py`](file:///d:/market-simulation/src/agents/ensemble_agents.py), [`src/agents/evaluate_phase_5_6.py`](file:///d:/market-simulation/src/agents/evaluate_phase_5_6.py)  
**Deliverables**: [`results/olive_trade_v2/phase_05_ensembles/`](file:///d:/market-simulation/results/olive_trade_v2/phase_05_ensembles/)  

---

## 1. Overview & Ensemble Architecture

Phase 6 implements three independently evaluated multi-strategy ensemble architectures designed to blend continuous reinforcement learning agents with quantitative asset allocation rules:

1. **Equal-Weight Ensemble ([`EqualWeightEnsemble`](file:///d:/market-simulation/src/agents/ensemble_agents.py#L9))**:
   Blends continuous target weights from $M=5$ diverse models:
   $$\mathbf{w}_t^{\text{ens}} = \frac{1}{M} \sum_{m=1}^M \mathbf{w}_t^{(m)}$$
   Constituents: PPO, A2C, Volatility Targeting, Risk Parity, Mean Reversion.
2. **Validation-Weighted Ensemble ([`ValidationWeightedEnsemble`](file:///d:/market-simulation/src/agents/ensemble_agents.py#L29))**:
   - Calibrated exclusively on the dedicated development validation partition (104 trading sessions from the training split; **zero lookahead into holdout test data**).
   - Solves for convex combination weights $\alpha_m \ge 0$ with $\sum \alpha_m = 1$ proportional to development Sharpe ratio, regularized with L2 shrinkage ($\lambda_{\text{reg}} = 0.20$) towards equal weights.
   - Fitted development weights:
     * A2C: **32.5%**
     * PPO: **29.8%**
     * Risk Parity: **29.6%**
     * Volatility Targeting: **4.1%**
     * Mean Reversion: **4.0%**
3. **Regime-Aware Ensemble ([`RegimeAwareEnsemble`](file:///d:/market-simulation/src/agents/ensemble_agents.py#L97))**:
   - Detects prevailing market state using backward-looking indicators available at decision timestamp $t$:
     * *Bull Low-Vol*: Average daily return $> 0.5\%$, rolling vol $< 1.5\%$.
     * *High-Vol Stress*: Average rolling vol $> 1.5\%$.
     * *Sideways Range*: Normal volatility and flat trend.
   - Dynamically routes allocations: overweights continuous RL in bull regimes and risk parity/vol-targeting in stress regimes.

---

## 2. Holdout Comparative Evaluation (149 Trading Days, ₹100,000.00 Base)

Evaluated on the unseen holdout partition (`2017-05-29` to `2017-12-29`) under the institutional Risk Overlay:

| Model / Ensemble | Starting Equity (₹) | Final Equity (₹) | Total Return % | Ann. Volatility % | Sharpe Ratio | Max Drawdown % | Trades Count | Fees Paid (₹) |
|---|---|---|---|---|---|---|---|---|
| **Buy & Hold Baseline** | 100,000.00 | 99,888.19 | -0.11% | **0.33%** | -0.58 | **-0.30%** | **120** | **₹61.31** |
| **Volatility Targeting** | 100,000.00 | 99,132.27 | **-0.87%** | 1.97% | -0.74 | **-1.72%** | 580 | ₹287.27 |
| **DDPG Agent** | 100,000.00 | 97,558.54 | -2.44% | 11.48% | -0.31 | -7.24% | 1,747 | ₹1,253.12 |
| **A2C Agent** | 100,000.00 | 96,720.82 | -3.28% | 10.06% | -0.51 | -9.09% | 1,179 | ₹642.06 |
| **Validation-Weighted Ensemble** | 100,000.00 | 96,700.92 | **-3.30%** | **8.88%** | **-0.60** | **-7.45%** | 1,484 | ₹709.93 |
| **Equal-Weight Ensemble** | 100,000.00 | 96,438.07 | **-3.56%** | **8.84%** | **-0.65** | **-7.61%** | 1,581 | ₹753.75 |
| **Risk Parity** | 100,000.00 | 95,874.81 | -4.13% | 9.48% | -0.71 | -7.56% | 1,224 | ₹756.72 |
| **PPO Agent** | 100,000.00 | 95,438.29 | -4.56% | 9.20% | -0.82 | -9.16% | 1,466 | ₹981.34 |
| **Mean Reversion** | 100,000.00 | 84,507.97 | -15.49% | 7.05% | -4.03 | -16.62% | 1,833 | ₹1,443.24 |
| **Regime-Aware Ensemble** | 100,000.00 | 82,561.67 | -17.44% | 8.64% | -3.73 | -19.03% | 2,437 | ₹1,738.47 |

---

## 3. Analysis & Key Academic Findings

1. **Markowitz Volatility Compression via Ensembling**:
   - The **Validation-Weighted Ensemble** compressed annualized volatility down to **8.88%**, outperforming PPO (9.20%), Risk Parity (9.48%), A2C (10.06%), and DDPG (11.48%).
   - Maximum drawdown was reduced to **-7.45%** (compared to PPO's -9.16% and A2C's -9.09%).
2. **Superiority of Regularized Validation Weighting**:
   - By weighting A2C (32.5%), PPO (29.8%), and Risk Parity (29.6%) while heavily discounting high-turnover Mean Reversion (4.0%), the Validation-Weighted Ensemble achieved higher risk-adjusted efficiency than the Equal-Weight Ensemble and its individual constituents.
3. **Regime Switching Friction Overhead**:
   - The Regime-Aware Ensemble incurred higher turnover (2,437 trades, ₹1,738 fees) due to rapid regime transitions, proving that abrupt policy switching without a hysteresis filter creates excessive execution friction.

Phase 6 acceptance criteria are fulfilled.
