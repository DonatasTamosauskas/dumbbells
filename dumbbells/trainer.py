import random
from itertools import count

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class Trainer:
    def __init__(
        self, dataset, agent, offline_update=10, batch_size=32, save_every=100
    ):
        self.dataset = dataset
        self.agent = agent
        self.offline_update = offline_update
        self.batch_size = batch_size

        self.ep_durations = []
        self.rewards = []
        self.losses = []
        self.save_every = save_every

    def _pre_fill_memory(self):
        """Pre-fill the memory with random actions"""
        while self.dataset.memory_size > len(self.dataset):
            self._play_episode()

        self.agent.reset_steps()

    def _play_episode(self):
        prev_state = self.dataset.reset()
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
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
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

            if ep % self.save_every == 0:
                self.agent.q.save_weights(
                    "model/weights"
                    + str(ep)
                    + str(time.strftime("%H-%M-%S", time.gmtime()))
                )
            self.ep_durations.append(time_steps)
            self.rewards.append(max_reward.item())
