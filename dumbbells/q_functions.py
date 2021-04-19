from __future__ import annotations
from typing import Union

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


class DnnQFunction(BaseQFunction):
    def __init__(
        self,
        arch: torch.nn.Module,
        gamma: float,
        optim: Union[torch.optim, None] = None,
    ):
        self.arch = arch
        self.gamma = gamma
        self.optimizer = (
            torch.optim.RMSprop(self.arch.parameters(), lr=1e-5)
            if optim is None
            else optim
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

    def train(self, states, actions, rewards, q_next_states):
        # TODO: Manage the special cases of end states. The pytorch example sets them to 0 value
        self.arch.train()

        state_action_values = self.arch(states).gather(1, actions)
        expected_state_action_values = (q_next_states * self.gamma) + rewards

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.arch.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def copy_weights(self):
        return self.arch.state_dict()

    def update(self, q_function: DnnQFunction):
        self.arch.load_state_dict(q_function.copy_weights())

    def eval(self):
        self.arch = self.arch.eval()
        return self
