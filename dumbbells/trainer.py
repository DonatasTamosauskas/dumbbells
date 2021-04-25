import random
from itertools import count

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, dataset, agent, offline_update=10, batch_size=32):
        self.dataset = dataset
        self.agent = agent
        self.offline_update = offline_update
        self.batch_size = batch_size

        self.ep_durations = []
        self.rewards = []
        self.losses = []

    def _pre_fill_memory(self):
        """Pre-fill the memory with random actions"""
        while self.dataset.memory_size > len(self.dataset):
            self._play_episode()

    def _play_episode(self):
        prev_state = torch.tensor([self.dataset.reset()], dtype=torch.float32)
        max_reward = None
        done = False

        for time_step in count():
            action = self.agent.action(prev_state.unsqueeze(0))
            action = action[0][0].item()
            prev_state, reward, done = self.dataset.step(action)

            if max_reward is None or max_reward < reward:
                max_reward = reward

            if done:
                return time_step, reward

    def train(self, episodes):
        self._pre_fill_memory()
        create_dl = lambda: iter(
            torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=True
            )
        )
        dl = create_dl()

        for ep in tqdm(range(episodes)):
            time_steps, max_reward = self._play_episode()

            try:
                batch = next(dl)
            except StopIteration:
                dl = create_dl()
                batch = next(dl)

            loss = self.agent.train_q(*batch)
            self.losses.append(loss)

            if ep % self.offline_update:
                self.agent.update_offline()

            self.ep_durations.append(time_steps)
            self.rewards.append(max_reward.item())
