
from pathlib import Path
import pandas as pd


class HistoricalMarketFeed:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "processed"
                / "selected_stocks.csv"
            )

        self.data = pd.read_csv(data_path, parse_dates=["TIMESTAMP"])
        self.data = self.data.sort_values(
            ["TIMESTAMP", "SYMBOL"]
        ).reset_index(drop=True)

        self.dates = sorted(self.data["TIMESTAMP"].unique())
        self.current_step = 0

    def reset(self):
        """Restart historical replay from the first date."""
        self.current_step = 0
        return self.get_current_market()

    def get_current_market(self):
        """Return all selected stocks for the current date."""
        if self.current_step >= len(self.dates):
            return None

        current_date = self.dates[self.current_step]
        day_data = self.data[
            self.data["TIMESTAMP"] == current_date
        ]

        return {
            row["SYMBOL"]: {
                "date": row["TIMESTAMP"],
                "open": float(row["OPEN"]),
                "high": float(row["HIGH"]),
                "low": float(row["LOW"]),
                "close": float(row["CLOSE"]),
                "volume": int(row["TOTTRDQTY"]),
                "turnover": float(row["TOTTRDVAL"]),
                "return": (
                    None if pd.isna(row["RETURN"])
                    else float(row["RETURN"])
                ),
            }
            for _, row in day_data.iterrows()
        }

    def step(self):
        """Advance to the next historical trading day."""
        self.current_step += 1
        return self.get_current_market()

    @property
    def is_finished(self):
        return self.current_step >= len(self.dates)
