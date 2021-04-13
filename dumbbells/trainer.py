from itertools import count

import torch


class Trainer:
    def __init__(self, dataset, agent, batch_size=32):
        self.dataset = dataset
        self.agent = agent

        self.ep_durations = []

    def _pre_fill_memory(self):
        """Pre-fill the memory with random actions"""
        pass

    def train(self, episodes):
        prev_state = self.dataset.get_state()
        done = False

        for ep in range(episodes):
            for time_step in count():
                action = self.agent.action(
                    torch.tensor([prev_state], dtype=torch.float32)
                )
                # TODO: Make all actions tensors (now the random ones produce integers)
                action = action[0][0].item()
                state, reward, done, _ = self.dataset.step(action)
                self.dataset.push_mem(prev_state, action, reward, state)
                prev_state = state

                if done:
                    self.ep_durations.append(time_step)
                    prev_state = self.dataset.reset()
                    break
