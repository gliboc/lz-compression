from math import erf, sqrt
import matplotlib.pyplot as plt
import numpy as np

def phi_normal(x):
    'Cumulative distribution for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def lz78_normal(mean, variance):
    'Gives cumulative distribution function for M_n from LZ78 if it was standard normal'
    return lambda x : phi_normal( (x - mean) / variance )


def print_normal(phi=None):
    'Prints a cumulative distribution function on an interval'
    x1 = -10
    x2 = 10

    if phi is None:
        f = phi_normal

    x = np.linspace(x1, x2, 100)
    y = [f(v) for v in x]

    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    print_normal()
