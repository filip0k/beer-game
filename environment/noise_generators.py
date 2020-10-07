import numpy as np


def generate_normal_noise(length=1):
    return np.random.normal(0, 1, length)


def generate_uniform_noise(low, high, length=1):
    return np.random.randint(low, high, length)
