import random
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent


class SumTree:
    """Binary tree data structure where parent node is sum of children nodes."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data: Any):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with proportional sampling."""

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 5000,
        epsilon_per: float = 1e-5,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon_per = epsilon_per
        self.frame = 1
        self.max_priority = 1.0

    def get_beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        p = self.max_priority ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size: int) -> Tuple[Tuple, np.ndarray, np.ndarray]:
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_memory = []
        priorities = np.empty((batch_size,), dtype=np.float64)

        segment = self.tree.total() / batch_size
        self.frame += 1
        beta = self.get_beta()

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities[i] = p
            b_idx[i] = idx
            b_memory.append(data)

        sampling_probabilities = priorities / max(self.tree.total(), 1e-8)
        is_weights = np.power(self.tree.size * sampling_probabilities, -beta)
        is_weights /= max(is_weights.max(), 1e-8)

        states, actions, rewards, next_states, dones = zip(*b_memory)
        batch = (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )
        return batch, b_idx, np.asarray(is_weights, dtype=np.float32)

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(tree_indices, td_errors):
            p = (abs(float(err)) + self.epsilon_per) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.size


class DuelingQNetwork(nn.Module):
    """Dueling Neural Network architecture with separate Value and Advantage streams."""

    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 128):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared representation feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # State-Value Stream: V(s) in R^1
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Action-Advantage Stream: A(s, a) in R^{|A|}
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Centered dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class DoubleDuelingDQNAgent(BaseAgent):
    """Double Dueling Deep Q-Network Agent with Prioritized Experience Replay (PER)."""

    ACTIONS = ("BUY", "HOLD", "SELL")

    def __init__(
        self,
        state_size: int = 10,
        action_size: int = 3,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_sync_freq: int = 100,
        seed: int = 42,
    ):
        super().__init__(
            name="DoubleDuelingDQN",
            config={
                "state_size": state_size,
                "action_size": action_size,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "epsilon_min": epsilon_min,
                "batch_size": batch_size,
                "memory_size": memory_size,
            },
        )
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
        self.target_sync_freq = target_sync_freq

        self.device = torch.device("cpu")
        self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # For per-sample IS weighting
        self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        self.training_steps = 0

    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """Evaluation prediction (epsilon=0)."""
        return self.act(observation, training=False)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy exploration."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=-1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store transition in Prioritized Replay Buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Perform one Double Dueling DQN update with PER importance sampling."""
        if len(self.memory) < self.batch_size:
            return None

        (states, actions, rewards, next_states, dones), tree_indices, is_weights = self.memory.sample(
            self.batch_size
        )

        s_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        r_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_s_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        w_tensor = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)

        # Current Q estimates
        current_q = self.policy_net(s_tensor).gather(1, a_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selected by policy_net, evaluated by target_net
            best_next_actions = self.policy_net(next_s_tensor).argmax(dim=-1, keepdim=True)
            target_next_q = self.target_net(next_s_tensor).gather(1, best_next_actions).squeeze(1)
            target_q = r_tensor + self.gamma * target_next_q * (1.0 - d_tensor)

        # Compute TD error for PER priority update
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(tree_indices, td_errors)

        # Weighted loss
        elementwise_loss = self.loss_fn(current_q, target_q)
        weighted_loss = torch.mean(elementwise_loss * w_tensor)

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        # Synchronize target network
        if self.training_steps % self.target_sync_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(weighted_loss.item())

    def save(self, path: str):
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

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", 0.05)
        self.training_steps = checkpoint.get("training_steps", 0)
