import gym
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as Ds


class Dataset(Ds):
    """Dataset containing information about the environment"""

    def __init__(self, game, memory_size):
        """Constructor, with the environment and memory_size specified

        Args:
            game: String containing the name of the environment to be created. Must be a valid environment. A list of
                  valid environments can be found at "https://gym.openai.com/envs/#classic_control", or with
                  gym.envs.registry.all()
            memory_size: Max images we want to store at any time.

        Returns: None
        """

        self.env = gym.make(game)
        self.memory_size = memory_size
        self.memory = []
        self.frames = []
        self.position = 0

        # For the environment "MountainCar-v0", there are 3 actions available:
        # 0: Accelerate to the left
        # 1: Don't accelerate
        # 2: Accelerate to the right
        self.action_space = self.env.action_space

        # For the environment "MountainCar-v0", there are 2 observations, presented in the following order:
        # Car Position: Ranges from -1.2 to 0.6
        # Car Velocity: Ranges from -0.07 to 0.07
        # The car starts at some position between -0.6 to -0.4
        self.state_space = self.env.reset()

        # For the environment "MountainCar-v0", there are 2 possible rewards:
        # Reward of 0 is awarded if the agent has reached the flag (position 0.5)
        # Reward of -1 is awarded if the agent is at a position < 0.5
        self.reward_space = [-1, 0]

    def get_state(self):
        """Return the current state of the game (the observations)

        Args: None

        Returns:
            Current state of the environment.
        """
        # For the environment "MountainCar-v0", this would return (Car_position, Car_velocity)
        return self.state_space

    def step(self, action):
        """Take the given action in the environment and update the memory

        Args:
            action: Action the agent will take in the environment

        Returns:
            (State, Reward, Done, {}): The updated state, the reward, whether the game is done, and {}.
        """
        prev_state = self.get_state()
        # Take the action
        result = self.env.step(action)
        # Update the state and memory
        self.state_space = result[0]
        self.push_mem(prev_state, action, result[1], self.get_state())
        # Store this frame to produce a gif later on
        self.frames.append(self.env.render(mode="rgb_array"))
        # Return the result of the action
        return result

    def push_mem(self, prev_state, action, reward, next_state):
        """Helper function to push an image and label to memory

        Args:
            prev_state: The previous state
            action: Action that was taken
            reward: Reward given for taking that action at that state
            next_state: Current state as a result of taking that action

        Return: None
        """
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.position] = (prev_state, action, reward, next_state)
        self.position = (self.position + 1) % self.memory_size

    def __getitem__(self, idx):
        """Returns the image and label stored in memory at the given index

        Args:
            idx: Valid index given the length of the memory

        Returns:
            (State, Action, Reward, Resulting_state): An image and corresponding label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.memory[idx]

    def __len__(self):
        """Finds how many image and label pairs are currently stored in memory

        Args: None

        Returns:
            Number of image / label pairs currently stored
        """
        return len(self.memory)

    def reset(self):
        """Resets the environment without clearing the memory

        Args: None

        Returns:
            Observations after resetting (the state_space)
        """
        result = self.env.reset()
        # Update the current state space
        self.state_space = result[0]
        return result

    def save_frames_as_gif(self, path="./", filename="gym_animation.gif"):
        """Produces a gif of the every action that has been taken by the agent on the environment from the
        beginning until called

        Args:
            path: Optional, file path where the gif will be saved relative to the directory calling this method.
                Default is the same directory as the caller.
            filename: Optional, file name of the gif to be saved. Default is "gym_animation.gif"

        Returns:
            None (though it produces a saved gif)

        """
        self.env.close()
        # Mess with this to change frame size
        plt.figure(
            figsize=(self.frames[0].shape[1] / 72.0, self.frames[0].shape[0] / 72.0),
            dpi=72,
        )

        patch = plt.imshow(self.frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(self.frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(self.frames), interval=50
        )
        anim.save(path + filename, writer="imagemagick", fps=60)
