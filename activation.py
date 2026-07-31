import numpy as np


def LeakyReLU(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def LeakyReLU_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)