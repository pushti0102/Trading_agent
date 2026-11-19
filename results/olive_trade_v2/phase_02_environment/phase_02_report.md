# OliveTrade 2.0 — Phase 2 Unified Trading Interface Report

**Date**: October 28, 2026  
**Module**: [`src/environment/gym_trading_env.py`](file:///d:/market-simulation/src/environment/gym_trading_env.py)  
**Deliverables**: [`results/olive_trade_v2/phase_02_environment/`](file:///d:/market-simulation/results/olive_trade_v2/phase_02_environment/)  

---

## 1. Overview & Architectural Design

Phase 2 replaces the legacy ad-hoc stepping logic with a standard Gymnasium-compliant interface ([`GymTradingEnv`](file:///d:/market-simulation/src/environment/gym_trading_env.py)). The environment wraps historical multi-asset data feeds, feature extractors, and the friction execution engine while enforcing rigorous quantitative constraints.

### Core Architecture:
- **Gymnasium Standard**:
  - `reset(seed=None, options=None) -> (obs, info)`
  - `step(action) -> (obs, reward, terminated, truncated, info)`
- **Observation Normalization**:
  - [`ObservationNormalizer`](file:///d:/market-simulation/src/environment/gym_trading_env.py#L11-L45) fits $(\mu, \sigma)$ exclusively on the 70% chronological training partition (346 dates). Frozen statistics are saved in `normalizer_params.json` and applied to test environments to ensure zero data leakage.
- **Dual Action Modes**:
  1. `continuous`: Bounded target portfolio weight vector $\mathbf{w}_t \in [0, 1]^N$.
  2. `discrete`: Preserves discrete single-stock or multi-stock action set $\{0=\text{BUY}, 1=\text{HOLD}, 2=\text{SELL}\}$.

---

## 2. Feasible Action Projection & Portfolio Allocation

Target portfolio weights are projected into feasible share quantities under strict institutional risk controls:
1. **Long-Only Constraint**: $w_i \ge 0$.
2. **Cash Reserve Floor**: $\sum_{i=1}^N w_i \le 1.0 - c_{\text{reserve}}$ (where $c_{\text{reserve}} = 5\%$).
3. **Single-Stock Concentration Cap**: $w_i \le 0.20$ ($20\%$ maximum equity in any single asset).
4. **Execution Priority**: Sells are filled first to generate liquidity, followed by Buys scaled to available spendable cash.
5. **Discrete Integer Shares**: Target capital values $V \cdot w_i$ are converted to integer shares at next Open $P_{\text{open}, t+1}$, accounting for half-spread ($1.0\%$), market impact ($0.0005 \times Q$), and transaction fees ($0.1\%$).

---

## 3. Unit Test Validation

All 7 unit and integration tests in [`tests/test_gym_env.py`](file:///d:/market-simulation/tests/test_gym_env.py) passed:
- `test_gym_env_spaces_discrete`: Validates discrete action shape `Discrete(3)` and single-stock state vector $(10,)$.
- `test_gym_env_spaces_continuous`: Validates continuous action vector $(40,)$ and flattened observation vector $(400,)$.
- `test_cash_conservation_and_inventory`: Proves cash decreases by exact trade value plus fees and inventory updates correctly.
- `test_cash_reserve_floor_enforced`: Proves portfolio cash never drops below the 5% cash reserve floor.
- `test_position_concentration_cap_enforced`: Confirms single-stock weights are bounded by the 20% cap.
- `test_observation_normalizer_strictly_fitted_on_train`: Verifies normalization and serialization without test set contamination.
- `test_chronological_advancement_and_termination`: Confirms termination occurs exactly when historical dates are exhausted.

---

## 4. Smoke Experiment Summary

- **Holdout Period**: 149 trading sessions (May 29, 2017 to Dec 29, 2017).
- **Initial Capital**: ₹100,000.00.
- **Final Portfolio Equity**: ₹102,492.00 (+2.49% net return).
- **Executed Trades**: 133 rebalancing orders.
- **Total Fees Paid**: ₹60.12.
- **Minimum Cash Recorded**: ₹81,886.48 (substantially above the ₹5,000 reserve floor).

Phase 2 acceptance criteria are fulfilled.
