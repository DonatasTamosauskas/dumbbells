from typing import Union

import torch
import torch.nn.functional as F


class BaseQFunction:
    def predict(self, state):
        raise NotImplementedError()

    def max_expected_reward(self, state):
        raise NotImplementedError()

    def train(state, action, reward, q_next_state):
        raise NotImplementedError()

    def update(self, q_function: BaseQFunction):
        raise NotImplementedError()
    

class DnnQFunction(BaseQFunction):
    def __init__(self, arch: torch.nn.Module, gamma: float, optim: Union[torch.optim, None] = None):
        self.arch = arch
        self.gamma = gamma
        self.optimizer = torch.optim.RMSprop(self.arch.parameters()) if optim is None else optim

    def predict(self, states):
        with torch.no_grad():
            return self.arch(states).max(dim=-1)[1].view(1,1)

    def max_expected_reward(self, states):
        with torch.no_grad():
            return self.arch(states).max(dim=-1)[0]

    def train(self, states, actions, rewards, q_next_states):
        # TODO: Manage the special cases of end states. The pytorch example sets them to 0 value

        state_action_values = self.arch(states).gather(1, actions)
        expected_state_action_values = (q_next_states * self.gamma) + rewards

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def copy_weights(self):
        return self.arch.state_dict()

    def update(self, q_function: DnnQFunction):
        self.arch.load_state_dict(q_function.copy_weights())

