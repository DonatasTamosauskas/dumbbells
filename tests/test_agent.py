import gym
import torch

from dumbbells.agent import Agent


def test_init():
    """Tests initialization of Agent"""

    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    reward_space = [-1, 0]

    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200

    current_state = env.reset()
    myQ = q_functions(current_state)
    my_agent = Agent(
        myQ, action_space, state_space, reward_space, eps_start, eps_end, eps_decay
    )

    assert my_agent.state_space == env.observation_space
    assert my_agent.action_space == env.action_space
    assert my_agent.reward_space == [-1, 0]
    assert my_agent.eps_start == eps_start
    assert my_agent.eps_end == eps_end
    assert my_agent.eps_decay == eps_decay
    assert my_agent.steps_done == 0

    # Technically an optional argument, but worth checking nonetheless
    assert my_agent.verbose == False


def test_q_action():
    """Tests that action can be taken by Q Function"""

    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    reward_space = [-1, 0]

    eps_start = 0
    eps_end = 2
    eps_decay = 1

    current_state = env.reset()
    my_q = q_functions(current_state)
    my_agent = Agent(
        my_q, action_space, state_space, reward_space, eps_start, eps_end, eps_decay
    )

    for t in range(200):
        my_action = my_agent.action(current_state).item()
        assert (my_q.old_state == current_state).all()
        current_state, reward, done, __ = env.step(my_action)
        if done:
            break

    assert my_agent.steps_done == 200


def test_random_action():
    """Tests that action can be taken randomly"""

    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    reward_space = [-1, 0]

    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200

    current_state = env.reset()
    my_q = q_functions(current_state)
    my_agent = Agent(
        my_q, action_space, state_space, reward_space, eps_start, eps_end, eps_decay
    )

    for t in range(200):
        my_action = my_agent.random_action().item()
        current_state, reward, done, __ = env.step(my_action)
        if done:
            break

    assert my_agent.steps_done == 200


class q_functions:
    """Temporary Q-Function for testing. Returns constant action of 1"""

    def __init__(self, current_state):
        self.old_state = current_state
        self.last_action = torch.tensor([[1]], dtype=torch.int)

    def predict(self, current_state):
        self.old_state = current_state
        return self.last_action

    def eval(self):
        return self
