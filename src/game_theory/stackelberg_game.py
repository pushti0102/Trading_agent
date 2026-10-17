
class StackelbergGame:
    """One-step Stackelberg game for leader-follower order decisions."""

    def __init__(self, max_order_size=100):
        self.max_order_size = int(max_order_size)

    def leader_action(self, observation):
        """Leader buys undervalued stocks and sells overvalued holdings."""
        actions = {}

        for symbol, state in observation["stocks"].items():
            price = state["price"]
            fundamental = state["fundamental_value"]
            inventory = state["inventory"]

            if price < fundamental * 0.99:
                actions[symbol] = ("BUY", min(5, self.max_order_size))
            elif price > fundamental * 1.01 and inventory >= 5:
                actions[symbol] = ("SELL", 5)
            else:
                actions[symbol] = ("HOLD", 0)

        return actions

    def follower_action(self, observation, leader_actions):
        """Follower observes leader's proposed actions and responds."""
        actions = {}

        for symbol, state in observation["stocks"].items():
            price_return = state["return"]
            inventory = state["inventory"]

            leader_side, _ = leader_actions[symbol]

            if leader_side == "BUY" and price_return >= 0:
                actions[symbol] = ("BUY", 5)
            elif leader_side == "SELL" and inventory >= 5:
                actions[symbol] = ("SELL", 5)
            elif price_return > 0.01:
                actions[symbol] = ("BUY", 5)
            elif price_return < -0.01 and inventory >= 5:
                actions[symbol] = ("SELL", 5)
            else:
                actions[symbol] = ("HOLD", 0)

        return actions
