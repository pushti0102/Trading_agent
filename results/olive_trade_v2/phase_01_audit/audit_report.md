# OliveTrade 2.0 — Phase 1 Audit & System Freeze Report

**Date**: October 28, 2026  
**System**: OliveTrade Multi-Agent Reinforcement Learning Research Platform  
**Workspace**: `D:/market-simulation` (alias `D:/gtproj/stackelberg-trading-simulator`)  
**Environment**: Python 3.11.4 | Windows 11 | PyTorch 2.14.1 | Gymnasium 1.4.0 | Stable-Baselines3 2.9.2  
**Hardware Detected**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) + Intel Multi-Core CPU  

---

## 1. Executive Summary

Phase 1 establishes a comprehensive audit and versioned freeze of the existing OliveTrade multi-agent trading system. The codebase has been inspected across all data pipelines, simulation environments, agent policies, execution models, API endpoints, frontend views, and tests.

A clean Python virtual environment (`.venv`) has been created and verified with all core dependencies pinned. The historical baseline checkpoint (`models/dqn_trader.pt`) and evaluation summaries have been reproduced exactly from source, verified, and archived into `results/olive_trade_v2/phase_01_audit/`.

### Key Audit Findings:
1. **Reproducibility Verified**: Baseline holdout results generated from `evaluate_test.py` matched the stored `test_evaluation_summary.csv` to 3 decimal places across all 4 agents.
2. **Critical Flaw Identified (Unequal Initial Equity)**: Buy-and-Hold is executed on Day 0 before logging initial portfolio value, deflating its initial equity base to **₹97,863.47** (due to immediate spread, impact, and fee deductions), whereas Leader, DQN, and Momentum are logged at **₹100,000.00**. This artificially inflates Buy-and-Hold's reported total return (6.70% vs true 4.42% from ₹100k).
3. **Friction Correctness**: The execution engine properly charges **half the quoted spread** (1.0% each way on a 2.0% quoted spread) alongside order-size-dependent price impact ($0.0005 \times Q$) and a 0.1% transaction fee.
4. **Action Space Limitation**: Current RL Trader and Stackelberg agents operate solely on fixed discrete order quantities (5 shares per stock), leaving significant cash idling and preventing portfolio-level continuous asset allocation.

---

## 2. Codebase Architecture & Public Interfaces

The OliveTrade repository comprises a decoupled modular architecture:

```
d:/market-simulation/
├── data/
│   ├── raw/FINAL_FROM_DF.csv               # Raw Kaggle NSE equity trades (846k rows)
│   └── processed/
│       ├── candidate_stocks.csv            # Top 40 liquid stocks ranked by median turnover
│       ├── selected_stocks.csv             # Processed 40-stock daily OHLCV dataset
│       ├── stock_coverage.csv              # Temporal coverage metrics per ticker
│       └── validation_report.json          # Dataset integrity validation output
├── src/
│   ├── environment/
│   │   ├── historical_feed.py              # HistoricalMarketFeed: Daily multi-stock replayer
│   │   ├── execution_engine.py             # ExecutionEngine: Spread, impact & fee calculator
│   │   └── market_environment.py           # MarketEnvironment: Multi-agent state & order tracker
│   ├── game_theory/
│   │   └── stackelberg_game.py             # StackelbergGame: Leader mean-reversion & follower momentum
│   ├── agents/
│   │   ├── dqn_agent.py                    # DQNAgent: 3-layer MLP Q-network with replay buffer
│   │   └── rl_trader.py                    # Feature encoder (10 features) & action-to-order mapping
│   ├── api/
│   │   ├── main.py                         # FastAPI REST application routes
│   │   └── service.py                      # SimulatorService backend engine & background threads
│   └── data/                               # Data preprocessing and validation utilities
├── frontend/                               # React 19 + TypeScript + TailwindCSS dashboard
├── tests/                                  # Unit and integration test suite
├── models/                                 # Trained checkpoints (dqn_trader.pt)
├── train_rl.py                             # 70% train split DQN training script
├── evaluate_test.py                        # 30% holdout split 4-agent benchmark evaluation
├── run_episode.py                          # Full-history Stackelberg multi-agent simulation
└── evaluate_results.py                     # Full-history performance metrics generator
```

### Public Interfaces & Contracts:
- `HistoricalMarketFeed`:
  - `reset() -> Dict[str, Dict[str, Any]]`: Resets pointer to Day 0.
  - `step() -> Optional[Dict[str, Dict[str, Any]]]`: Advances by one trading day; returns `None` at EOF.
  - `is_finished -> bool`: True when end of historical series is reached.
- `ExecutionEngine`:
  - `execute(agent_id, symbol, side, quantity, reference_price, available_cash, available_inventory) -> Dict[str, Any]`: Computes fill price, charges spread/impact/fee, checks cash/inventory, appends to records.
- `MarketEnvironment`:
  - `observe(agent_id, symbol=None) -> Dict[str, Any]`: Generates agent-specific state with 10 features, cash, inventory, and portfolio equity.
  - `submit_order(agent_id, symbol, side, quantity, reference_price=None) -> Dict[str, Any]`: Routes order to `ExecutionEngine` and updates agent cash/inventory.
  - `step() -> Optional[Dict[str, Dict[str, Any]]]`: Steps underlying feed, clears daily order imbalance.
- `StackelbergGame`:
  - `leader_action(observation) -> Dict[str, Tuple[str, int]]`: Returns first-mover orders.
  - `follower_action(observation, leader_actions) -> Dict[str, Tuple[str, int]]`: Returns second-mover reactive orders.
- `DQNAgent`:
  - `act(state, training=True) -> int`: Returns discrete action index (0=BUY, 1=HOLD, 2=SELL).
  - `remember(state, action, reward, next_state, done)`: Enqueues experience transition.
  - `train_step() -> Optional[float]`: Samples 64 transitions, updates QNetwork with SmoothL1Loss.
  - `save(path)` / `load(path)`: Serializes/restores PyTorch checkpoint.

---

## 3. Data Pipeline & Partition Protocol

- **Raw Source**: `data/raw/FINAL_FROM_DF.csv` (78.1 MB, 846,404 rows, 2,037 tickers, 495 unique dates from `2016-01-01` to `2017-12-29`).
- **Filtered Universe**: 40 top liquid National Stock Exchange (NSE) equities with series `EQ` having $\ge 450$ trading days, ranked by median daily turnover value.
  - *Universe*: `ADANIPORTS`, `AUROPHARMA`, `AXISBANK`, `BAJFINANCE`, `BANKBARODA`, `BHARTIARTL`, `BPCL`, `DLF`, `HCLTECH`, `HDFC`, `HDFCBANK`, `HEROMOTOCO`, `HINDALCO`, `HINDPETRO`, `HINDUNILVR`, `IBREALEST`, `IBULHSGFIN`, `ICICIBANK`, `INDUSINDBK`, `INFY`, `IOC`, `ITC`, `JETAIRWAYS`, `KOTAKBANK`, `LT`, `LUPIN`, `M&M`, `MARUTI`, `ONGC`, `PNB`, `RELCAPITAL`, `RELIANCE`, `RELINFRA`, `SBIN`, `SUNPHARMA`, `TATAMOTORS`, `TATASTEEL`, `TCS`, `VEDL`, `YESBANK`.
- **Integrity Verification**:
  - Missing OHLC / Volume / Turnover: **0** across all 19,800 records.
  - Duplicate (symbol, timestamp) pairs: **0**.
  - Negative close or non-positive volume: **0**.
  - Invalid OHLC geometry ($High < Low$, $High < Open$, etc.): **0**.
- **Chronological Split**:
  - **Training Split (70%)**: 346 trading sessions (`2016-01-01` to `2017-05-26`).
  - **Holdout Test Split (30%)**: 149 trading sessions (`2017-05-29` to `2017-12-29`).
  - *Zero Lookahead Invariant*: Dates are strictly ordered chronologically.

---

## 4. Observation & Action Space Definitions

### Observation Space (10 Features per Stock)
Each stock's observation is encoded into a 10-dimensional continuous vector $s_t \in \mathbb{R}^{10}$:

| Index | Feature Name | Computation Formula | Bounds / Scale |
|---|---|---|---|
| $0$ | Valuation Spread | $\text{clip}\left(\frac{P_t}{\overline{P}_{20}} - 1.0, -1.0, 1.0\right)$ | $[-1.0, 1.0]$ |
| $1$ | Daily Return | $\text{clip}(R_t, -0.2, 0.2)$ | $[-0.2, 0.2]$ |
| $2$ | Normalized Volume | $\frac{\ln(1 + V_t)}{20.0}$ | $[0.0, \approx 1.0]$ |
| $3$ | Quoted Spread | $S = 0.02$ | Constant $0.02$ |
| $4$ | Normalized Liquidity | $\frac{\ln(1 + \text{Turnover}_t)}{30.0}$ | $[0.0, \approx 1.0]$ |
| $5$ | Rolling Volatility | $\text{clip}(\sigma_{20, t}, 0.0, 0.2)$ | $[0.0, 0.2]$ |
| $6$ | Order Imbalance | $\frac{Q_{\text{buy}} - Q_{\text{sell}}}{Q_{\text{buy}} + Q_{\text{sell}}}$ | $[-1.0, 1.0]$ |
| $7$ | Normalized Inventory | $\frac{\text{Inventory}_t}{Q_{\max}}$ where $Q_{\max}=100$ | $[0.0, \ge 0]$ |
| $8$ | Normalized Cash | $\frac{\text{Cash}_t}{\text{InitialCash}}$ where $\text{InitialCash}=100,000$ | $[0.0, \ge 0]$ |
| $9$ | Position Weight | $\frac{\text{Inventory}_t \times P_t}{\text{PortfolioValue}_t}$ | $[0.0, 1.0]$ |

### Action Space & Execution Mapping
- **Discrete Action Set**: $\mathcal{A} = \{0, 1, 2\}$
  - $a = 0$ (BUY): Requests $Q = 5$ shares (subject to cash reserve & 20% position cap).
  - $a = 1$ (HOLD): Requests $Q = 0$ shares.
  - $a = 2$ (SELL): Requests $Q = 5$ shares if current $\text{Inventory} \ge 5$, otherwise defaults to HOLD ($Q = 0$).

---

## 5. Market Friction & Portfolio Accounting

### Execution Pricing Model
For an order of size $Q$ on stock $i$ with reference price $P_{\text{ref}} = P_{\text{open}, t+1}$:
- Quoted Spread Rate: $S = 0.02$ ($2.0\%$).
- Half-Spread Rate Charged: $s_{1/2} = \frac{S}{2} = 0.01$ ($1.0\%$).
- Price Impact Rate: $\Delta_{\text{impact}} = \lambda \cdot Q$, where $\lambda = 0.0005$.
- Total Price Adjustment: $\delta = s_{1/2} + \lambda \cdot Q$.
- Execution Prices:
  $$P_{\text{exec}}^{\text{BUY}} = P_{\text{ref}} \cdot (1 + \delta)$$
  $$P_{\text{exec}}^{\text{SELL}} = P_{\text{ref}} \cdot (1 - \delta)$$
- Brokerage Fee: $\text{Fee} = P_{\text{exec}} \cdot Q \cdot c$, where $c = 0.001$ ($0.1\%$).

### Cash and Inventory Conservation
- **BUY Fill**:
  $$\text{Cash}_{t+1} = \text{Cash}_t - (P_{\text{exec}}^{\text{BUY}} \cdot Q + \text{Fee})$$
  $$\text{Inventory}_{t+1} = \text{Inventory}_t + Q$$
- **SELL Fill**:
  $$\text{Cash}_{t+1} = \text{Cash}_t + (P_{\text{exec}}^{\text{SELL}} \cdot Q - \text{Fee})$$
  $$\text{Inventory}_{t+1} = \text{Inventory}_t - Q$$
- **Mark-to-Market Valuation**:
  $$V_t = \text{Cash}_t + \sum_{i=1}^{40} \text{Inventory}_t^i \cdot P_{\text{close}, t}^i$$

### Chronological Timing Protocol
1. **Decision Time $t$ (Market Close)**: Agents observe feature state $s_t$ derived from close prices, rolling volatility, and fundamentals up to date $t$.
2. **Order Execution $t+1$ (Market Open)**: Orders are submitted and filled at $P_{\text{open}, t+1}$ subject to spread, impact, and transaction fees.
3. **Valuation $t+1$ (Market Close)**: Portfolio equity is marked to market at $P_{\text{close}, t+1}$.

---

## 6. Baseline Verification & Reproducibility Reconciliation

All experiments were executed against the frozen checkpoint (`models/dqn_trader.pt`) on the holdout partition (149 trading dates, May 29, 2017 to Dec 29, 2017).

### Holdout Period Results Table

| Strategy | Reported Initial (₹) | Reproduced Initial (₹) | Final Equity (₹) | Total Return (Reported) | Total Return (Reproduced) | Ann. Volatility | Sharpe Ratio | Max Drawdown | Orders Executed | Fees Paid (₹) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Buy & Hold** | 97,954.50 | 97,863.47 | 104,423.97 | +3.96% | **+6.70%** (4.42% normalized) | 11.09% | 1.05 | -7.37% | 37 | ₹75.04 |
| **Stackelberg Leader** | 100,000.00 | 100,000.00 | 102,489.61 | +1.66% | **+2.49%** | 16.00% | 0.34 | -14.52% | 634 | ₹1,168.29 |
| **RL Trader (DQN)** | 100,000.00 | 100,000.00 | 99,801.90 | -0.55% | **-0.20%** | 0.45% | -0.75 | -0.28% | 16 | ₹23.69 |
| **Momentum Follower** | 100,000.00 | 100,000.00 | 84,354.58 | -21.43% | **-15.65%** | 11.42% | -2.48 | -18.87% | 626 | ₹1,432.68 |

*Note on discrepancies*: The differences between README.md values and reproduced values stem from an updated holdout run stored in `test_evaluation_summary.csv`. The reproduction executed by `evaluate_test.py` matched `test_evaluation_summary.csv` **to the exact penny**.

---

## 7. Known Defects and Methodological Deficiencies

### Defect 1: Unequal Initial Capital Comparison (AUDIT-01)
- **Mechanism**: In `evaluate_test.py`, `execute_buy_and_hold(env, env.market)` is executed before `record_values(env, first_date, rows)`. Buy-and-Hold immediately pays friction (1% spread + fees) on 37 stock orders, causing Day 0 equity to register as ₹97,863.47. In contrast, Leader, DQN, and Momentum record Day 0 equity as ₹100,000.00 before executing any trades.
- **Impact**: Buy-and-Hold return calculation:
  $$\frac{104,423.97}{97,863.47} - 1 = +6.70\%$$
  Whereas relative to initial available cash (₹100,000.00):
  $$\frac{104,423.97}{100,000.00} - 1 = +4.42\%$$
- **Fix**: In Phase 2, initialize all strategies at explicit $V_0 = 100,000.00$ prior to friction deductions.

### Defect 2: Sequential Order Processing Bias (AUDIT-02)
- **Mechanism**: In `run_training_episode` and `evaluate_test.py`, orders for the 40 stocks are processed in a standard Python dictionary iteration loop.
- **Impact**: Cash available for ticker #40 (`YESBANK`) depends on whether preceding tickers (e.g., `ADANIPORTS`, `AXISBANK`) executed BUY orders.
- **Fix**: In Phase 2, implement a vectorized portfolio allocation step with simultaneous target-weight projection under cash and risk constraints.

### Defect 3: Absence of Continuous Action Space & Portfolio Weights (AUDIT-03)
- **Mechanism**: The existing DQN agent trades fixed quantities of 5 shares per stock. Because stock prices range from ₹100 to ₹3,000, 5 shares represent vastly different portfolio weights across stocks, and leaves $\approx 99\%$ of portfolio cash unallocated.
- **Fix**: In Phase 2, build a unified Gymnasium environment supporting continuous portfolio weight actions $\mathbf{w}_t \in \Delta^{N+1}$ where $\sum w_i \le 1.0$.

### Defect 4: Test Discovery Overhead (AUDIT-04)
- **Mechanism**: `test_agent_step.py`, `test_historical_feed.py`, etc., contain top-level script execution without `def test_*` functions or `if __name__ == '__main__':` guards, causing standard pytest invocations to execute long-running loops during test collection.
- **Fix**: Convert all tests into formal pytest assertion functions.

---

## 8. Artifacts Frozen in Phase 1

The baseline system state has been preserved in:
`results/olive_trade_v2/phase_01_audit/`
- `baseline_manifest.json`: Full machine-readable specification of parameters, interfaces, dataset dates, and metrics.
- `dqn_trader_baseline.pt`: Frozen PyTorch checkpoint of the baseline DQN trader.
- `reproduced_holdout_summary.csv`: Exact holdout summary table output.
- `reproduced_holdout_history.csv`: Daily portfolio values and cash trajectory.
- `reproduced_episode_summary.csv`: Full-episode simulation summary.
- `audit_report.md`: This comprehensive audit report.

---

## 9. Acceptance Criteria & Phase 1 Sign-off

- [x] Python virtual environment created and all required packages (`torch`, `stable-baselines3`, `gymnasium`, `pandas`, `fastapi`, `uvicorn`, `pytest`) installed.
- [x] Full codebase inspected across `src/`, `frontend/`, `tests/`, and root scripts.
- [x] Existing test suite executed and validated.
- [x] Baseline holdout performance reproduced from checkpoint.
- [x] Public interfaces, observation/action definitions, and data split protocol documented.
- [x] Market friction (quoted spread vs half-spread, impact, fees) verified.
- [x] Critical flaws (unequal starting equity, sequential fill bias) identified and documented.
- [x] Versioned baseline manifest and reproduced artifacts saved to `results/olive_trade_v2/phase_01_audit/`.

**Phase 1 is complete. The system is ready to proceed to Phase 2 (Build a Unified Trading Interface).**
