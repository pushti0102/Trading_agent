# Stackelberg Trading Simulator — Runbook

## Prerequisites
- Activate your existing virtual environment.
- Ensure `data/processed/selected_stocks.csv` exists.
- The CSV should contain `TIMESTAMP`, `SYMBOL`, `OPEN`, `HIGH`, `LOW`, `CLOSE`,
  `TOTTRDQTY`, `TOTTRDVAL`, and `RETURN` columns.

## Main workflow
1. Train DQN only on the first 70% of dates:
   ```powershell
   python train_rl.py
   ```
   Outputs: `models/dqn_trader.pt`, `rl_training_results.csv`.

2. Evaluate the saved DQN on the final 30% and compare it with Stackelberg
   leader/follower and an equal-budget buy-and-hold baseline:
   ```powershell
   python evaluate_test.py
   ```
   Outputs: `test_portfolio_history.csv`, `test_evaluation_summary.csv`.

3. Run the full-history Stackelberg simulation:
   ```powershell
   python run_episode.py
   ```

4. Summarize the full-history episode:
   ```powershell
   python evaluate_results.py
   ```

## Interpretation notes
- The execution engine simulates spread, impact, and transaction costs; these are
  assumptions, not real historical order-book measurements.
- Features use rolling historical data. Test dates are kept out of DQN updates.
- Buy-and-hold is an equal-budget basket constrained by the simulated maximum
  order size, so it is a practical baseline rather than a perfect index replica.
- Returns on a single historical split are not proof of future profitability.
- Do not tune settings against the holdout period repeatedly; reserve it for
  final evaluation and use a separate validation period for model selection.
