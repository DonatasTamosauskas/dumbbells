class Agent:
    def __init__(self, Q, action_space, state_space, reward_space, verbose=False):
        self.Q = Q
        self.action_space = action_space
        self.state_space = state_space
        self.reward_space = reward_space

        self.verbose = verbose
        if self.verbose:
            print("Agent Initialized\n.................")

    # Agent performs random action from Action_Space
    def random_action(self):
        rand_action = self.action_space.sample()
        if self.verbose:
            print("Random Action: ", rand_action)
        return rand_action

    # Agent performs action from Q-Function
    def action(self, current_state):
        best_action = self.Q.predict(current_state)
        if self.verbose:
            print("Best Action: ", best_action)
        return best_action
