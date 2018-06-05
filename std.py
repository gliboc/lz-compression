from math import log
import scipy
import numpy as np

def stationary_distribution(M):
    """Computes a stationary distribution for a given Markov chain M.

    Args:
     M (int matrix): The Markov chain.

    Returns:
     (float array): A stationary distribution of M.
    """

    S, U = scipy.linalg.eig( M.T )
    p = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    p = p / np.sum(p)

    if 0:
     print("p times M", np.dot( p, M ))
     print("p", p)

    return p


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

       h += p[i] * M[i, j] * log( M[i, j], 2)

    return (-h)


def omega(M):
    return (1 - M[1,1]) * M[0,1]

def beta(M):
    s = 0
    p00 = M[0, 0]
    p11 = M[1, 1]
    p01 = M[0, 1]
    p10 = M[1, 0]

    s += p00 * p11 * (log (p00)) ** 2 * (log (p11)) ** 2

    s -= p01 * p10 * (log (p01)) ** 2 * (log (p10)) ** 2

    return s

def pi_q_psi(M, p):
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
    s -= (2 * pi_q_psi(M, p)) / o - h ** 2

    return s
