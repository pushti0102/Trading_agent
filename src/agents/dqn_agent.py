
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Neural network that estimates Q-values for trading actions."""

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """DQN agent with experience replay and epsilon-greedy exploration."""

    ACTIONS = ("BUY", "HOLD", "SELL")

    def __init__(
        self,
        state_size,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000,
        seed=42,
    ):
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size != len(self.ACTIONS):
            raise ValueError("This agent supports BUY, HOLD, and SELL")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cpu")
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.memory = deque(maxlen=memory_size)
        self.training_steps = 0

    def act(self, state, training=True):
        """Return an action index: BUY=0, HOLD=1, SELL=2."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if state.size != self.state_size:
            raise ValueError(
                f"Expected {self.state_size} state values, got {state.size}"
            )

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            return int(self.policy_net(state_tensor).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store one transition in replay memory."""
        self.memory.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def train_step(self):
        """Perform one DQN update. Returns loss, or None if memory is small."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )

        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True
        )
        if (
            checkpoint["state_size"] != self.state_size
            or checkpoint["action_size"] != self.action_size
        ):
            raise ValueError("Checkpoint architecture does not match this agent")

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
