from markov import *


def Hi(M, i):
    """Computes the entropy of branching from state i."""

    r = 0

    for j in [0, 1]:

        r += M[i, j] * log(M[i, j], 2)

    return r


def H0(M):
    return Hi(M, 0)


def H1(M):
    return Hi(M, 1)


def sigma_i(M, i):
    """Computes H^3 * sigma_i^2, from Neininger's paper."""

    pi_i = M[1 - i, i] / (M[0, 1] + M[1, 0])

    r = pi_i * M[i, 0] * M[i, 1]

    r *= (
        log(M[i, 0], 2) - log(M[i, 1], 2) + (H1(M) - H0(M)) / (M[0, 1] + M[1, 0])
    ) ** 2

    return r


def sigma2_H3(M):
    """Computes H^3 * sigma^2, from Neininger's paper."""

    return sigma_i(M, 0) + sigma_i(M, 1)

