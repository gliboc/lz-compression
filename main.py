from __future__ import division
import numpy as np # learn more: https://python.org/pypi/np
from math import log
import scipy

debug = True

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

   S, U = scipy.linalg.eig( M.T )
   p = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
   p = p / np.sum(p)

   if debug:
     print("p times M", np.dot( p, M ))
     print("p", p)

   return p


def entropy(M, p=None):
   """Computes the entropy of a known Markov chain by computing a stationary distribution.

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

       h += p[i] * M[i, j] * log( M[i, j] )

   return (-h)


def h_2(M, p=None, h=None):
    """Computes the second derivative of lambda, taken in s=-1 (see Average profile of the Lempel-Ziv parsing scheme for a Markovian source).

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

    n = len(m)

    t1 = 0
    t2 = 0
    t3 = 0

    for i in range(n):
        for j in range(n):

            t1 += p[i] * M[i, j] * (log ( M[i, j] )) ** 2

    for i in range(n):
        for j in range(n):

            t2 += log ( p[i] * M[i, j] ) * p[i] * M[i, j]

    t2 *= 2

    for i in range(n):
        for j in range(n):

            t3 += p[i] * log( p[i] )

    t3 *= 2 * h

    return t1 + t2 + t3


def variance(M)



def test_entropy():

   Ms = [markov_chain(i) for i in range(2, 10)]
   ents = [entropy(M) for M in Ms]

   for i in range(len(Ms)):
     print("The Markov chain:")
     raw_input(Ms[i])
     print("Its entropy:")
     raw_input(ents[i])


def psi(n):
    """Returns the Psi vector: a 1-D array of size n filled with ones.

    Args:
        n (int): Size of the Markov chain.

    Returns:
        (int array): The Psi vector
    """

    return np.ones(n)


import cmath

def markov_source(M, f, n):
  """Outputs a word of size n from a Markov source (M, f)

  Args:
    M (int matrix): Markov chain.
    f (int row): State assignment function.
    n (int): Length of message.

  Returns:
    digit list: The generated message.
  """

  current_state = 0
  word = []

  for _ in range(n):

    transition_proba = np.random.rand(1)

    next_state = 0
    proba_stack = M[current_state, 0]

    while transition_proba > proba_stack:

      next_state += 1
      proba_stack += M[current_state, next_state]

    current_state = next_state
    word.append(f[current_state])

  return word


# Some examples

M = markov_chain(2)
f = [0,1]
w1 = markov_source(M, f, 5)
w2 = markov_source(M, f, 10)
w3 = markov_source(M, f, 20)
w4 = markov_source(M, f, 30)

def word_generator(M, f, n):
  return lambda : markov_source(M, f, n)

word_10 = word_generator(M, f, 10)


def compress(word):
  """Compression of a word using LZ78.

  Args:
    word (digit list): The word to compress.

  Returns:
    string list: The list of phrases used in LZ78.
  """

  s = set()
  phrases = [""]
  current_prefix = ""

  for digit in word:
    digit = str(digit)

    if current_prefix + digit in s:
      current_prefix += digit

    else:
      s.add(current_prefix + digit)
      phrases.append(current_prefix + digit)
      current_prefix = ""

  if current_prefix != "":
    #print("The last phrase is incomplete:", current_prefix)
    phrases.append(current_prefix)

  return phrases


import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.kde import gaussian_kde
from numpy import linspace


def redundancy_histograms(random_markov=False):
  """Makes a simulation of the redundancy distribution for different word length values,
  and prints the corresponding histograms.
  """
  i = 100
  length_values = [4*i, 10*i, 20*i, 100*i]
  n_exp = 200

  if not random_markov:
    p_a = 0.9
    M = np.matrix([[p_a, 1-p_a], [1-p_a, p_a]])
    h = - p_a * log (p_a) - (1-p_a) * log (1-p_a)
    f = [0, 1]

  else:
    N = np.random.randint(2, 11)
    M = markov_chain(N)
    h = entropy(M)
    f = state_fun(N)
    raw_input("Using a random Markov chain of size" + str(N))
    raw_input(M)
    print("Its state function f is:")
    raw_input(f)
    print("Its entropy is:")
    raw_input(h)

  rates = []
  figs, axs = plt.subplots(2, len(length_values), tight_layout=True)

  for i, n in enumerate(length_values):
    n_exp = n # USE A MORE POWERFUL PC
    print("Simulation with words of size", n)
    print("Doing %d experiments" % n_exp)

    word_gen = word_generator(M, f, n)

    l = [word_gen() for _ in range(n_exp)]
    c = [compress(w) for w in l]
    r = [len(x) / n - h for x in c]

    if 0:
      print("These are some word examples:")
      _ = raw_input(l[:10])

      print("And their codes:")
      _ = raw_input(c[:10])

      print("And their rates:")
      _ = raw_input(r[:10])

    kde = gaussian_kde( r )
    dist_space = linspace( min(r), max(r), 100 )

    axs[0][i].hist(r, bins=50, facecolor='g')
    axs[1][i].plot( dist_space, kde(dist_space) )

    for k in [0, 1]:
      axs[k][i].set_xlabel('Rates')
      axs[k][i].set_ylabel('Probability')
      axs[k][i].grid(True)

  print("Done")
  plt.show()



if __name__ == "__main__":
  print("Printing histograms:")
  redundancy_histograms()
  redundancy_histograms(random_markov=True)
