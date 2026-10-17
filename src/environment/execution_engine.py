
class ExecutionEngine:
    def __init__(
        self,
        transaction_cost=0.001,
        max_order_size=100,
        impact_coefficient=0.0005,
        spread_rate=0.02,
    ):
        self.transaction_cost = float(transaction_cost)
        self.max_order_size = int(max_order_size)
        self.impact_coefficient = float(impact_coefficient)
        self.spread_rate = float(spread_rate)
        self.records = []

    def execute(
        self,
        agent_id,
        symbol,
        side,
        quantity,
        reference_price,
        available_cash,
        available_inventory,
    ):
        side = str(side).upper()

        if side not in ("BUY", "SELL"):
            return {"status": "REJECTED", "reason": "Invalid side"}

        if (
            not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or quantity <= 0
        ):
            return {
                "status": "REJECTED",
                "reason": "Quantity must be a positive integer",
            }

        if quantity > self.max_order_size:
            return {
                "status": "REJECTED",
                "reason": "Maximum order size exceeded",
            }

        if reference_price <= 0:
            return {"status": "REJECTED", "reason": "Invalid reference price"}

        if self.spread_rate < 0 or self.impact_coefficient < 0:
            return {
                "status": "REJECTED",
                "reason": "Spread and impact must be non-negative",
            }

        # Half the quoted spread is charged on each side.
        half_spread = self.spread_rate / 2
        impact_rate = self.impact_coefficient * quantity

        total_adjustment = half_spread + impact_rate

        if side == "BUY":
            execution_price = reference_price * (1 + total_adjustment)
        else:
            execution_price = reference_price * (1 - total_adjustment)

        if execution_price <= 0:
            return {
                "status": "REJECTED",
                "reason": "Invalid execution price",
            }

        trade_value = execution_price * quantity
        fee = trade_value * self.transaction_cost

        if side == "BUY" and available_cash < trade_value + fee:
            return {"status": "REJECTED", "reason": "Insufficient cash"}

        if side == "SELL" and available_inventory < quantity:
            return {
                "status": "REJECTED",
                "reason": "Insufficient inventory",
            }

        record = {
            "status": "EXECUTED",
            "agent": agent_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "reference_price": float(reference_price),
            "execution_price": float(execution_price),
            "trade_value": float(trade_value),
            "fee": float(fee),
            "spread_rate": float(self.spread_rate),
            "impact_rate": float(impact_rate),
        }

        self.records.append(record)
        return record
