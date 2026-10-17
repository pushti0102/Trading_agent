
import pandas as pd

INPUT_FILE = "episode_results.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values(["agent", "date"])

def evaluate(group):
    values = group["portfolio_value"].astype(float)
    daily_returns = values.pct_change().dropna()

    total_return = (values.iloc[-1] / values.iloc[0]) - 1

    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = (
            daily_returns.mean()
            / daily_returns.std(ddof=1)
        ) * (252 ** 0.5)

    drawdown = values / values.cummax() - 1
    max_drawdown = drawdown.min()

    return pd.Series({
        "Initial Value": values.iloc[0],
        "Final Value": values.iloc[-1],
        "Total Return (%)": total_return * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown * 100,
    })


summary = df.groupby("agent").apply(
    evaluate, include_groups=False
)

print(summary.round(3))
summary.to_csv("evaluation_summary.csv")
print("\nSaved: evaluation_summary.csv")
