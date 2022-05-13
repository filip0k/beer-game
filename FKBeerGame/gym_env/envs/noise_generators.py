import numpy as np


def generate_normal_noise(length=1):
    return np.asscalar(np.random.normal(0, 1, length))


def generate_uniform_noise(low, high, length=1):
    return np.random.randint(low, high, length)


def generate_standard_noise(i):
    if i >= 4:
        return 8
    else:
        return 4
