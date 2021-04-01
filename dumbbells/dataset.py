import gym
from torch.utils.data import Dataset, DataLoader

class Dataset:

    # Constructor, with the environment and memory_size specified
    def __init__(self, game, memory_size):
        self.env = gym.make(game)
        self.memory_size = memory_size

        # For the environment "MountainCar-v0", there are 3 actions available:
        # 0: Accelerate to the left
        # 1: Don't accelerate
        # 2: Accelerate to the right
        self.action_space = self.env.action_space

        # For the environment "MountainCar-v0", there are 2 observations, presented in the following order:
        # Car Position: Ranges from -1.2 to 0.6
        # Car Velocity: Ranges from -0.07 to 0.07
        # The car starts at some position between -0.6 to -0.4
        self.state_space = self.env.state_space

        # For the environment "MountainCar-v0", there are 2 possible rewards:
        # Reward of 0 is awarded if the agent has reached the flag (position 0.5)
        # Reward of -1 is awarded if the agent is at a position < 0.5
        self.reward_space = -1

    # Return the current state of the game (the observations)
    def get_state(self):
        return self.state_space

    # Given an action, take that action and returns the following tuple:
    # State, Reward, Done, {}
    def step(self, action):
        # Take the action
        result = self.env.step(action)
        # Update the state
        self.state_space = result[0]
        # Return the result of the action
        return result

    # Returns image and label -> action, reward, next state
    def __getitem__(self):
        return
