from __future__ import division
import numpy # learn more: https://python.org/pypi/numpy
from math import log

def markov_chain(n):
	"""Generate a random markov chain with n states
  
  Args:
    n (int): The number of states

  Returns:
    (n, n) int matrix: The markov chain
  """

	matrix = numpy.random.rand(n, n)
	return matrix / matrix.sum(axis=1)[:, None]


def state_fun(n):
	"""Assigns a character (0 or 1) to each of n states.

  Args:
    n (int): The number of states

  Returns:
    digit list: The list of assignments. l[i] is the digit of state i.
  """

	return numpy.random.randint(0, 2, n)


def stationary_distribution(M):
   """Computes a stationary distribution for a given Markov chain M.

   Args:
     M (int matrix): The Markov chain.

   Returns:
     (float array): A stationary distribution of M.
   """
   
   Mt = numpy.transpose (M)
   p = numpy.linalg.eig (Mt)
   
   return p
   

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
    
    transition_proba = numpy.random.rand(1)
    
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
import numpy as np
from matplotlib import colors
from scipy.stats.kde import gaussian_kde
from numpy import linspace


def redundancy_histograms():
  """Makes a simulation of the redundancy distribution for different word length values,
  and prints the corresponding histograms.
  """
  i = 100
  length_values = [4*i, 10*i, 20*i, 100*i]
  n_exp = 200

  p_a = 0.9
  M = numpy.matrix([[p_a, 1-p_a], [1-p_a, p_a]])
  f = [0, 1]
  h = - p_a * log (p_a) - (1-p_a) * log (1-p_a)

  rates = []
  figs, axs = plt.subplots(1, len(length_values), sharey=True, sharex=True, tight_layout=True)

  for i, n in enumerate(length_values):
    print("Simulation with words of size", n)

    word_gen = word_generator(M, f, n)

    if 1:
      l = [word_gen() for _ in range(n_exp)]
      print("These are some word examples:")
      _ = raw_input(l)

      c = [compress(w) for w in l]
      print("And their codes:")
      _ = raw_input(c)

      r = [len(x) / n - h for x in c]
      print("And their rates:")
      _ = raw_input(r)

    kde = gaussian_kde( r )
    dist_space = linspace( min(r), max(r), 100 )

    axs[i].hist(r, bins=50, facecolor='g')
    axs[i].plot( dist_space, kde(dist_space) )

    axs[i].set_xlabel('Rates')
    axs[i].set_ylabel('Probability')
    axs[i].grid(True)

  print("Done")
  plt.show()

if __name__ == "__main__":
  print("Printing histograms:")
  redundancy_histograms()
