"""Functions for simulating a Markov chain"""

import numpy as np
import scipy.linalg


def markov_chain(n):
    """Generate a random markov chain with n states.

    Args:
    n (int): The number of states

    Returns:
    (n, n) int matrix: The markov chain
    """

    matrix = np.random.rand(n, n)
    return matrix / matrix.sum(axis=1)[:, None]


def state_fun(n):
    """Assigns a character (0 or 1) to each of n states.

    Args:
    n (int): The number of states

    Returns:
    digit list: The list of assignments. l[i] is the digit of state i.
    """

    return np.random.randint(0, 2, n)


def stationary_distribution(M):
    """Computes a stationary distribution for a given Markov chain M.

    Args:
     M (int matrix): The Markov chain.

    Returns:
     (float array): A stationary distribution of M.
    """

    S, U = scipy.linalg.eig(M.T)
    p = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    p = p / np.sum(p)

    # if 0:
    #     print("p times M", np.dot(p, M))
    #     print("p", p)

    p01 = M[0, 1]
    p10 = M[1, 0]

    x = p10 / (p01 + p10)
    y = p01 / (p01 + p10)

    if abs(x - p[0]) > 0.001 or abs(y - p[1]) > 0.001:
        raise ValueError

    return [x, y]


def markov_source(M, n):
    """Outputs a word of size n from a Markov source (M, f)
    !! Now only works with chain  s of size 2 !!

    Args:
    M (int matrix): Markov chain.
    f (int row): State assignment function.
    n (int): Length of message.

    Returns:
    digit list: The generated message.
    """

    current_state = 0
    word = []
    probas = np.random.rand(n)

    for i in range(n):

        # next_state = 0
        proba_stack = M[current_state, 0]

        current_state = int(probas[i] > proba_stack)
        # while probas[i] > proba_stack:

        #  next_state += 1
        #  proba_stack += M[current_state, next_state]

        # current_state = next_state
        word.append(current_state)

    # print("Generated word")
    # input(word)
    return word


def markov_source2(M, n):
    """Generate a word in a string format"""

    c_state = "0"
    word = ""
    probas = np.random.rand(n)
    d = {"0": M[0, 0], "1": M[1, 0]}

    for i in range(n):
        c_state = str(int(probas[i] > d[c_state]))
        word += c_state

    return word



def word_generator(M, n):
    return lambda: markov_source(M, n)


from progress.bar import Bar

if __name__ == "__main__":
    M = markov_chain(2)
    bar = Bar("Processing", max=20)