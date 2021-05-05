from __future__ import annotations
from typing import Union, Optional

import torch
import torch.nn.functional as F


class BaseQFunction:
    """Implements the q-function"""

    def predict(self, states: torch.tensor) -> torch.tensor:
        """Predicts the next action

        Args:
            states: tensor batch representing the env states

        Returns:
            tensor batch of actions
        """

        raise NotImplementedError()

    def max_expected_reward(self, states: torch.tensor) -> torch.tensor:
        """Calculate the max expected rewards given states

        Args:
            states: tensor batch representing the env states

        Returns:
            tensor batch of max expected rewards
        """

        raise NotImplementedError()

    def train(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        q_next_states: torch.tensor,
    ):
        """Update model from a batch of actions

        Args:
            states: tensor batch of env states
            actions: tensor batch of actions taken
            rewards: tensor batch of rewards received
            q_next_states: tensor batch of max expected rewards from the next state
        """

        raise NotImplementedError()

    def update(self, q_function: BaseQFunction):
        """Update models from another q-function

        Args:
            q_function: the BaseQFunction to update from
        """

        raise NotImplementedError()

    def save_weights(self, path):
        """Saves model and optimizer weights into a file at the given path

        Args:
            path: Name of the file the model and optimizer weights will be saved to
        """

        raise NotImplementedError()

    def load_weights(self, path):
        """Loads model and optimizer weights given a filename

        Args:
            path: Name of the files the model and optimizer weights will be loaded from
        """

        raise NotImplementedError()


class DnnQFunction(BaseQFunction):
    def __init__(
        self,
        arch: torch.nn.Module,
        gamma: float,
        optim: Optional[torch.optim] = None,
        lr: Optional[float] = 1e-3,
    ):
        self.arch = arch
        self.gamma = gamma
        self.optimizer = (
            torch.optim.Adam(self.arch.parameters(), lr=lr) if optim is None else optim
        )

    def predict(self, states):
        self.arch.eval()

        with torch.no_grad():
            return self.arch(states).max(dim=-1)[1].view(-1, 1)

    def max_expected_reward(self, states):
        self.arch.eval()

        with torch.no_grad():
            ans = self.arch(states).max(dim=-1)[0].view(-1, 1)
            return ans

    def train(self, states, actions, rewards, q_next_states, dones):
        self.arch.train()

        state_action_values = self.arch(states).gather(1, actions)
        expected_state_action_values = (q_next_states * self.gamma) + rewards
        expected_state_action_values[dones] = rewards[dones]

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.arch.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.detach().item()

    def copy_weights(self):
        return self.arch.state_dict()

    def save_weights(self, path):
        torch.save(self.arch.state_dict(), path + "_model_weights.model")
        torch.save(self.optimizer.state_dict(), path + "_optimizer_weights.optimizer")

    def load_weights(self, path):
        self.arch.load_state_dict(torch.load(path + "_model_weights.model"))
        self.optimizer.load_state_dict(
            torch.load(path + "_optimizer_weights.optimizer")
        )

    def update(self, q_function: DnnQFunction):
        self.arch.load_state_dict(q_function.copy_weights())

    def eval(self):
        self.arch = self.arch.eval()
        return self
