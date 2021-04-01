

class BaseQFunction:
    def __init__(self):
        pass

    def predict(self, state):
        raise NotImplementedError()

    def max_expected_reward(self, state):
        raise NotImplementedError()

    def train(state, action, reward, q_next_state):
        raise NotImplementedError()

    def update(self, q_function: BaseQFunction):
        raise NotImplementedError()
    