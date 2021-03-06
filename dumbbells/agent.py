import random
from copy import deepcopy

import numpy as np
import torch


class Agent:
    def __init__(
        self,
        q,
        action_space,
        state_space,
        reward_space,
        eps_start,
        eps_end,
        eps_decay,
        verbose=False,
    ):
        """Initiates the Agent and sets the necessary variables
        Args:
            q:              Q-Function used to determine actions from states
            action_space:   Discrete(3)
                                0: Move Left
                                1: No action
                                2: Move Right
            state_space:    Box(Position, Velocity, (2,), float32)
                                Max:
                                    Position: 0.6
                                    Velocity: 0.07
                                Min:
                                    Position: -1.2
                                    Velocity: -0.07
            rewards_space:  Reward of 0 is given if objective is reached (position = 0.5), else
                                reward of -1 is given per timestep
            verbose:        Optional argument that toggles display of extra information to console

        Returns: None
        """
        # Environment and Q function
        self.q = q
        self.q_offline = deepcopy(q.eval())
        self.action_space = action_space
        self.state_space = state_space
        self.reward_space = reward_space

        # Epsilon
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        # Optional verbose output used for testing
        self.verbose = verbose
        if self.verbose:
            print("Agent Initialized\n.................")

    def reset_steps(self):
        self.steps_done = 0

    def random_action(self):
        """Selects a random action from the action_space
        Args: None

        Returns:
            rand_action:    Integer representing action in action_space
        """
        # Picks a random value from the action space
        rand_action = self.action_space.sample()
        self.steps_done += 1

        if self.verbose:
            print("Random Action: ", rand_action)
        return torch.tensor([[rand_action]])

    def action(self, current_state):
        """Selects the best action determined by the Q-Function
        Args:
            current_state:  Box containing Position and Velocity in state_space

        Returns:
            best_action:    Integer representing the best action given Q-Function and current_state
        """
        # Calculates the epsilon value based on steps taken and initial values
        eps_val = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

        # Checks whether to pick a random action or Q function
        if eps_val < random.random():
            best_action = self.q.predict(current_state)
            self.steps_done += 1
        else:
            best_action = self.random_action()

        if self.verbose:
            print("Best Action: ", best_action)
        return best_action

    def train_q(self, states, actions, rewards, next_states, dones):
        max_expected = self.q_offline.max_expected_reward(next_states)
        return self.q.train(states, actions, rewards, max_expected, dones)

    def update_offline(self):
        self.q_offline.update(self.q)
