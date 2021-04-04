import pytest
import torch

from dumbbells.q_functions import *


def test_import():
    assert True


def test_dqn_q_func_init():
    net = torch.nn.Sequential(torch.nn.Linear(1, 1))
    gamma = 0.5

    q_func = DnnQFunction(arch=net, gamma=gamma)

    assert q_func is not None


def test_dqn_q_func_predict():
    net = torch.nn.Sequential(torch.nn.Linear(5, 2))
    q_func = DnnQFunction(arch=net, gamma=0.5)
    data = torch.ones((10, 5))

    out = q_func.predict(data)

    assert out.shape == (data.shape[0], 1)


def test_dqn_q_func_max_expected_reward():
    net = torch.nn.Sequential(torch.nn.Linear(5, 2))
    q_func = DnnQFunction(arch=net, gamma=0.5)
    data = torch.ones((10, 5))

    out = q_func.max_expected_reward(data)

    assert out.shape == (data.shape[0], 1)
