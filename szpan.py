"""Expressions from the paper Szpan. & Jacquet"""

from math import log
import numpy as np
from markov import stationary_distribution, markov_chain


def entropy(M, p=None):
    """Computes the entropy of a known Markov chain by computing a
    stationary distribution.

    Args:
     M (float matrix): The Markov chain.
     [p] (float array): A previously computed stationary distribution.

    Returns:
      (float): The entropy of M.

    """

    if p is None:
        p = stationary_distribution(M)

    h = 0
    n = len(M)

    for i in range(n):
        for j in range(n):

            h += p[i] * M[i, j] * log(M[i, j])

    return -h


def h_2(M, p=None, h=None):
    r"""Computes the second derivative of lambda, taken in s=-1
    (see Average profile of the Lempel-Ziv parsing scheme for a Markovian source).

    Args:
      M (float matrix): The Markov chain.
      p (float array): A left eigenvector.
      [h] (float): The precomputed entropy of M.

    Returns:
      (float): \dot\dot lambda (-1)
    """

    if p is None:
        p = stationary_distribution(M)

    if h is None:
        h = entropy(M, p=p)

    n = len(M)

    t1 = 0
    t2 = 0
    t3 = 0

    for i in range(n):
        for j in range(n):

            t1 += p[i] * M[i, j] * log(M[i, j], 2) ** 2

    for i in range(n):
        for j in range(n):

            t2 += log(p[i], 2) * log(M[i, j], 2) * p[i] * M[i, j]

    t2 *= 2

    for i in range(n):
        for j in range(n):

            t3 += p[i] * log(p[i], 2)

    t3 *= 2 * h

    # print("h2's terms are %f, %f, %f" % (t1, t2, t3))
    return t1 + t2 + t3


def test_h2():
    print("Testing the h2 function.")

    print("On first order Markov chains:")

    for _ in range(10):

        M = markov_chain(2)
        print(M)
        print("Has entropy %f" % entropy(M))
        print("Has stationary distribution:")
        p = stationary_distribution(M)
        print(p)
        print("The stationary distribution entropy is:")
        print(sum([-x * log(x, 2) for x in p]))
        print("Its h2 is:")
        input(h_2(M))


def H(M):
    """Entropy using base 2 logarithm"""
    
    p = stationary_distribution(M)
    h = 0
    n = len(M)

    for i in range(n):
        for j in range(n):

            h += p[i] * M[i, j] * log(M[i, j], 2)

    return -h


def omega(M):
    """Certified"""
    return (1 - M[1, 1]) + M[0, 1]


def beta(M):
    """Certified"""
    s = 0

    p00 = M[0, 0]
    p11 = M[1, 1]
    p01 = M[0, 1]
    p10 = M[1, 0]

    s += p00 * p11 * (log(p00)) ** 2 * (log(p11)) ** 2
    s -= p01 * p10 * (log(p01)) ** 2 * (log(p10)) ** 2

    return s


def pi_q_psi(M, p):
    """Certified"""
    s = 0

    p00 = M[0, 0]
    p11 = M[1, 1]
    p01 = M[0, 1]
    p10 = M[1, 0]

    s += p[0] * p11 * log(p11)
    s -= p[1] * p10 * log(p10)
    s -= p[0] * p01 * log(p01)
    s += p[1] * p00 * log(p00)

    return s


def var(M):
    b = beta(M)
    p = stationary_distribution(M)
    o = omega(M)
    h = entropy(M)

    s = 0.

    s -= b / o
    s -= ((2 * pi_q_psi(M, p)) / o - h ** 2) / h ** 3

    return s


def psi(n):
    """Returns the Psi vector: a 1-D array of size n filled with ones.

    Args:
        n (int): Size of the Markov chain.

    Returns:
        (int array): The Psi vector
    """

    return np.ones(n)
