from itertools import count

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        dataset,
        agent,
        offline_update=10,
        steps_to_offline_update=128,
        batch_size=32,
        epsilon=1,
    ):
        # Environment
        self.dataset = dataset

        # Agent playing the game
        self.agent = agent

        # Offline Q function to determine the expected output
        self.steps_to_offline_update = steps_to_offline_update

        self.offline_update = offline_update
        self.counter = 128
        # How many frames we will sample from the memory_buffer when we want to train
        self.batch_size = batch_size

        # How likely are we to take a random action? Exploration vs Exploitation
        self.epsilon = epsilon

        # Lists to store additional information
        self.ep_durations = []
        self.rewards = []
        self.losses = []

    def _prefill_memory(self):
        index = 0
        for i in range(len(self.dataset)):
            state = self.dataset.reset()
            done = False
            while not done:
                action = self.agent.action(
                    state, self.dataset.action_space.n, epsilon=1
                )
                next_state, reward, done, _ = self.dataset.step(action)
                # self.agent.collect_experience([obs, A.item(), reward, next_state])
                state = next_state
                index += 1
                if index > len(self.dataset):
                    break

    def play_episode(self):
        state = self.dataset.reset()
        # Fresh environment, so fresh episode_reward and episode_loss!
        episode_reward, episode_loss = 0, 0
        for episode_time_steps in count(1):

            action = self.agent.action(state, self.epsilon)
            next_state, reward, done = self.dataset.step(action)
            # self.agent.collect_experience([obs, A.item(), reward, obs_next])

            episode_reward += reward

            if self.counter % self.steps_to_offline_update == 0:
                states, action, reward, next_states = self.dataset.get_sample(
                    self.batch_size
                )
                episode_loss += sum(
                    [
                        self.agent.train(
                            self.offline_update, states, reward, next_states
                        )
                        for j in range(4)
                    ]
                )

            state = next_state
            self.counter += 1
            if done:
                return episode_time_steps, episode_reward, episode_loss

    def train(self, episodes):
        self._prefill_memory()
        for episode in tqdm(range(episodes)):
            episode_time_steps, episode_reward, episode_loss = self.play_episode()

            if self.epsilon > 0.05:
                self.epsilon -= 1 / 5000

            # self.losses.append()
            self.ep_durations.append(episode_time_steps)
            self.rewards.append(episode_reward)
