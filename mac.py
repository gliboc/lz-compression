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

   if 0:
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

   for i in xrange(n):
     for j in xrange(n):

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

    n = len(M)

    t1 = 0
    t2 = 0
    t3 = 0

    for i in xrange(n):
        for j in xrange(n):

            t1 += p[i] * M[i, j] * (log ( M[i, j] )) ** 2

    for i in xrange(n):
        for j in xrange(n):

            t2 += log( p[i] ) * log( M[i, j] ) * p[i] * M[i, j]

    t2 *= 2

    for i in xrange(n):
        for j in xrange(n):

            t3 += p[i] * log( p[i] )

    t3 *= 2 * h

    print("h2's terms are %f, %f, %f" % (t1, t2, t3))
    return t1 + t2 + t3


def test_h2():
    print("Testing the h2 function.")

    print("On first order Markov chains:")

    for _ in xrange(10):

        M = markov_chain(2)
        print(M)
        print("Has entropy %f" % entropy(M))
        print("Has stationary distribution:")
        p = (stationary_distribution(M))
        print(p)
        print("The stationary distribution entropy is:")
        print(sum([-x * log(x) for x in p]))
        print("Its h2 is:")
        raw_input(h_2(M))


def Hi(M, i):
    """Computes the entropy of branching from state i."""

    r = 0

    for j in [0, 1]:

        r += M[i, j] * log( M[i, j] )

    return r

def H0(M):
    return Hi(M, 0)

def H1(M):
    return Hi(M, 1)


def sigma_i(M, i):
    """Computes H^3 * sigma_i^2, from Neininger's paper."""

    pi_i = M[1-i, i] / (M[0, 1] + M[1, 0])

    r = pi_i * M[i, 0] * M[i, 1]

    r *= (log( M[i, 0] ) - log( M[i, 1] ) + (H1(M) - H0(M)) / (M[0, 1] + M[1, 0])) ** 2

    return r

def sigma(M):
    """Computes H^3 * sigma^2, from Neininger's paper."""

    return sigma_i(M, 0) + sigma_i(M, 1)


def test_neininger():

    Ms = [markov_chain(2) for _ in xrange(100)]
    neins = [sigma(M) for M in Ms]
    ours = [(h_2(M) - entropy(M) ** 2) * entropy(M) ** 3 for M in Ms]

    cell_text = []

    for i in xrange(len(Ms)):
        cell_text.append(['%1.5f' % x for x in [neins[i], ours[i]]])

    columns = ['Neiningers H^3 * sigma^2', 'h2 - h^2']
    rows = [str((M[0,0], M[1, 0])) for M in Ms]

    plt.table (cellText=cell_text,
               rowLabels=rows,
               colLabels=columns)

    plt.plot(neins, color='r', label='Neininger')
    plt.plot(ours, color='g', label='Szpankowski')

    plt.legend()
    plt.title('Szpan vs Nein comparison')
    plt.show()


def test_entropy():

   Ms = [markov_chain(i) for i in xrange(2, 10)]
   ents = [entropy(M) for M in Ms]

   for i in xrange(len(Ms)):
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

  for _ in xrange(n):

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
from math import sqrt


def redundancy_histograms(random_markov=False):
  """Makes a simulation of the redundancy distribution for different word length values,
  and prints the corresponding histograms.
  """
  i = 100
  length_values = [4*i, 10*i, 20*i]

  if not random_markov:
    p_a = 0.9
    M = np.matrix([[p_a, 1-p_a], [1-p_a, p_a]])
    h = - p_a * log (p_a) - (1-p_a) * log (1-p_a)
    f = [0, 1]

  else:
    #N = np.random.randint(2, 11)
    N = 2
    M = markov_chain(N)
    h = entropy(M)
    #f = state_fun(N)
    f = [0, 1]
    raw_input("Using a random Markov chain of size" + str(N))
    raw_input(M)
    print("Its state function f is:")
    raw_input(f)
    print("Its entropy is:")
    raw_input(h)
    print("Its h2 is:")
    raw_input(h_2(M))
    print("h2-h^2 is:")
    raw_input(h_2(M)-h**2)

  rates = []
  figs, axs = plt.subplots(2, len(length_values), tight_layout=True)

  for i, n in enumerate(length_values):
    n_exp = n * 10  # USE A MORE POWERFUL PC
    print("Simulation with words of size", n)
    print("Doing %d experiments" % n_exp)

    word_gen = word_generator(M, f, n)

    var_coeff_Nein = sqrt (sigma(M) * n)
    var_coeff_Szpan = sqrt ( -(h_2(M) - (h**2)) * h**3 * n )


    l = [word_gen() for _ in xrange(n_exp)]
    c = [compress(w) for w in l]
    r_Nein = [(len(x) / n - h) / var_coeff_Nein  for x in c]
    r_Szpan = [(len(x) / n - h) / var_coeff_Szpan for x in c]

    if 0:
      print("These are some word examples:")
      _ = raw_input(l[:10])

      print("And their codes:")
      _ = raw_input(c[:10])

      print("And their rates:")
      _ = raw_input(r[:10])

    kde_Szpan = gaussian_kde( r_Szpan )
    kde_Nein = gaussian_kde( r_Nein )

    dist_space_Nein = linspace( min(r_Szpan), max(r_Szpan), 100 )
    dist_space_Szpan = linspace( min(r_Nein), max(r_Nein), 100 )

    bins = 20
    axs[0][i].hist(r_Nein, bins=30, color='r', label='Nein')
    axs[1][i].hist(r_Szpan, bins=30, color='g', label='Szpan')

    #axs[1][i].plot( dist_space_Nein, kde_Szpan(dist_space_Nein), color='r', label='Szpan' )
    #axs[1][i].plot( dist_space_Szpan, kde_Nein(dist_space_Szpan), color='g', label='Nein' )

    axs[0][i].legend()
    axs[1][i].legend()

    for k in [0, 1]:
      axs[k][i].set_xlabel('Rates')
      axs[k][i].set_ylabel('Probability')
      axs[k][i].grid(True)

  print("Done")
  plt.show()



if __name__ == "__main__":
  print("Printing histograms:")
  redundancy_histograms(random_markov=True)
