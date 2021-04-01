import gym

# TEMPORARY, ONLY USED TO MAKE THIS TEST WORK FOR NOW
import sys

sys.path.append("../dumbbells")
from dumbbells.Agent import Agent


def testQAgent():
    print("\nStarting testQAgent")

    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    reward_space = [-1, 0]

    current_state = env.reset()
    myQ = q_functions(current_state)
    # myAgent = Agent.Agent(myQ,action_space,state_space,reward_space,verbose = True)
    myAgent = Agent(myQ, action_space, state_space, reward_space, verbose=True)

    for t in range(10):
        # env.render()
        myAction = myAgent.action(current_state)
        current_state, reward, done, __ = env.step(myAction)
        if done:
            print("Finished after {} timesteps".format(t + 1))
            break
    env.close()
    print("Completed testQAgent")


def testRandAgent():
    print("\nStarting testRandAgent")

    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    reward_space = [-1, 0]

    current_state = env.reset()
    myQ = q_functions(current_state)
    # myAgent = Agent.Agent(myQ,action_space,state_space,reward_space,verbose = True)
    myAgent = Agent(myQ, action_space, state_space, reward_space, verbose=True)

    for t in range(10):
        # env.render()
        myAction = myAgent.random_action()
        current_state, reward, done, __ = env.step(myAction)
        if done:
            print("Finished after {} timesteps".format(t + 1))
            break
    env.close()
    print("Completed testRandAgent")


class q_functions:
    def __init__(self, current_state):
        self.old_state = current_state
        self.last_action = 1

    def predict(self, current_state):
        self.old_state = current_state
        return self.last_action


if __name__ == "__main__":
    testQAgent()
    testRandAgent()
    print("\n************************\nCompleted All Test\n************************\n")
