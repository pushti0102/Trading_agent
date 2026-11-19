# OliveTrade 2.0 — Phase 5 Portfolio Reward & Risk Management Report

**Date**: October 29, 2026  
**Module**: [`src/risk/risk_overlay.py`](file:///d:/market-simulation/src/risk/risk_overlay.py), [`src/environment/reward_functions.py`](file:///d:/market-simulation/src/environment/reward_functions.py)  
**Deliverables**: [`results/olive_trade_v2/phase_05_risk_management/`](file:///d:/market-simulation/results/olive_trade_v2/phase_05_risk_management/)  

---

## 1. Overview & Mathematical Formulations

Phase 5 implements configurable reward variants and an institutional risk overlay that projects raw agent target weights into mathematically feasible allocations under strict market friction constraints.

### Reward Function Variants:
- **Variant A (Net Log Return)**:
  $$R_t^{\text{log}} = \ln\left(\frac{V_{t+1}}{V_t}\right)$$
  Evaluates net portfolio equity $V_{t+1}$ marked at Close(t+1) after deducting execution half-spread (1.0%), quadratic/linear market impact ($0.0005 \times Q$), and commissions (0.1%) incurred at Open(t+1).
- **Variant B (Turnover-Penalized Log Return)**:
  $$R_t^{\text{turnover}} = \ln\left(\frac{V_{t+1}}{V_t}\right) - \lambda_{\text{turnover}} \sum_{i=1}^N |w_{i, t+1} - w_{i, t}|$$
  Directly penalizes portfolio churn, encouraging sparse and deliberate rebalancing.
- **Variant C (Downside Risk / Semi-Variance Penalty)**:
  $$R_t^{\text{downside}} = \ln\left(\frac{V_{t+1}}{V_t}\right) - \lambda_{\text{downside}} \left(\min(0, r_t)\right)^2$$
  Imposes an asymmetric penalty on portfolio losses, training agents to avoid left-tail drawdowns.

---

## 2. Institutional Risk Overlay Architecture

The [`PortfolioRiskOverlay`](file:///d:/market-simulation/src/risk/risk_overlay.py) guarantees:
1. **Long-Only Projection**: $w_i \ge 0$.
2. **Cash Reserve Floor**: $\sum_{i=1}^N w_i \le 1.0 - c_{\text{reserve}}$ (5% minimum cash floor).
3. **Single-Stock Concentration Cap**: $w_i \le 0.20$ (no constituent exceeds 20% of equity).
4. **Turnover Damping Constraint**:
   $$\Delta w_i = \text{clip}(w_i^{\text{target}} - w_i^{\text{curr}}, -\Delta w_{\max}, \Delta w_{\max})$$
   Smooths erratic reallocation and reduces execution costs.
5. **Drawdown Circuit Breaker**:
   If current peak-to-trough drawdown exceeds $-10\%$, equity exposure is automatically scaled down by 50% until recovery.
6. **Volatility Targeting**:
   Scales aggregate weight by $\min(1.0, \sigma_{\text{target}} / \hat{\sigma}_t)$ during market volatility spikes.

---

## 3. Empirical Risk Overlay Ablation Results

Evaluated on the 30% holdout test partition (149 trading dates, ₹100,000.00 initial capital):

| Risk Configuration | Initial Equity (₹) | Final Equity (₹) | Total Return % | Ann. Volatility % | Sharpe Ratio | Max Drawdown % | Trades Count | Fees Paid (₹) |
|---|---|---|---|---|---|---|---|---|
| **Full Risk Overlay** | 100,000.00 | 99,195.52 | **-0.80%** | **1.91%** | **-0.71** | **-1.68%** | **557** | **₹250.57** |
| **No Turnover Damping** | 100,000.00 | 98,363.50 | -1.64% | 2.20% | -1.27 | -2.31% | 600 | ₹382.63 |
| **No Volatility Target** | 100,000.00 | 99,195.52 | -0.80% | 1.91% | -0.71 | -1.68% | 557 | ₹250.57 |
| **No Circuit Breaker** | 100,000.00 | 99,195.52 | -0.80% | 1.91% | -0.71 | -1.68% | 557 | ₹250.57 |

### Key Takeaway:
Turnover damping halved the net capital loss from **-1.64% down to -0.80%** and cut friction fees by **35%** (from ₹382.63 to ₹250.57), verifying that friction-constrained action projection is critical for microstructural survival.

Phase 5 acceptance criteria are fulfilled.
