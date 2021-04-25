from copy import deepcopy
import random
import numpy as np
import torch


class Agent:
    def __init__(self, q, action_space):
        torch.manual_seed(1423)
        self.q = q
        self.q_offline = deepcopy(self.q)
        self.action_space = action_space
        self.counter = 1

    def action(self, state, epsilon):
        if epsilon > random.random():
            best_action = self.action_space.sample()
        else:
            with torch.no_grad():
                q_predict = self.q.arch(torch.from_numpy(state).float())
            Q, best_action = torch.max(q_predict, axis=0)
            best_action = best_action.item()
        return best_action

    def train(self, offline_update, state, reward, next_state):
        if self.counter % offline_update == 0:
            self.q_offline.arch.load_state_dict(self.q.arch.state_dict())

        with torch.no_grad():
            q_predicition = self.q_offline.arch(next_state)
        next_states, _ = torch.max(q_predicition, axis=1)
        loss = self.q.train(state, reward, next_states)

        self.counter += 1
        return loss
