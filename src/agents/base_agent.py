from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseAgent(ABC):
    """Common interface for all OliveTrade 2.0 trading agents."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def predict(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate action from current observation.

        Returns:
            Continuous target weights (N,) or discrete action vector/index.
        """
        pass

    def update(self, transition: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Optional learning step for trainable agents."""
        return None

    def save(self, path: str):
        """Serialize model weights and hyperparameters."""
        pass

    def load(self, path: str):
        """Restore model weights and hyperparameters."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration dictionary."""
        return {
            "name": self.name,
            **self.config
        }
