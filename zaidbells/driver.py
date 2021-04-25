from zaidbells.agent import Agent
from zaidbells.dataset import Dataset
from zaidbells.models import FcSimpleDqn
from zaidbells.q_functions import DnnQFunction
from zaidbells.trainer import Trainer

env = "CartPole-v1"
memory_buffer = 512  # 10000
gamma = 0.99

eps_start = 0.999
eps_end = 0.05
eps_decay = 200  # 200

episodes = 100000

dataset = Dataset(env, memory_buffer)
arch = FcSimpleDqn(dataset.state_space.shape[0], dataset.action_space.n)
q_func = DnnQFunction(arch.model, gamma)
agent = Agent(q_func, dataset.action_space)
trainer = Trainer(dataset, agent, batch_size=32, epsilon=1)
trainer.train(episodes)
print(sum(trainer.ep_durations) / len(trainer.ep_durations))
print(trainer.rewards)
