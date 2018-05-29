import numpy as np # learn more: https://python.org/pypi/np
from math import log
import scipy
from scipy.stats import norm

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

   for i in range(n):
     for j in range(n):

       h += p[i] * M[i, j] * log( M[i, j], 2)

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

    for i in range(n):
        for j in range(n):

            t1 += p[i] * M[i, j] * log ( M[i, j] , 2) ** 2

    for i in range(n):
        for j in range(n):

            t2 += log( p[i] , 2) * log( M[i, j], 2) * p[i] * M[i, j]

    t2 *= 2

    for i in range(n):
        for j in range(n):

            t3 += p[i] * log( p[i], 2)

    t3 *= 2 * h

    #print("h2's terms are %f, %f, %f" % (t1, t2, t3))
    return t1 + t2 + t3


def test_h2():
    print("Testing the h2 function.")

    print("On first order Markov chains:")

    for _ in range(10):

        M = markov_chain(2)
        print(M)
        print("Has entropy %f" % entropy(M))
        print("Has stationary distribution:")
        p = (stationary_distribution(M))
        print(p)
        print("The stationary distribution entropy is:")
        print(sum([-x * log(x, 2) for x in p]))
        print("Its h2 is:")
        input(h_2(M))


def Hi(M, i):
    """Computes the entropy of branching from state i."""

    r = 0

    for j in [0, 1]:

        r += M[i, j] * log( M[i, j], 2)

    return r

def H0(M):
    return Hi(M, 0)

def H1(M):
    return Hi(M, 1)


def sigma_i(M, i):
    """Computes H^3 * sigma_i^2, from Neininger's paper."""

    pi_i = M[1-i, i] / (M[0, 1] + M[1, 0])

    r = pi_i * M[i, 0] * M[i, 1]

    r *= (log( M[i, 0] , 2) - log( M[i, 1] , 2) + (H1(M) - H0(M)) / (M[0, 1] + M[1, 0])) ** 2

    return r

def sigma(M):
    """Computes H^3 * sigma^2, from Neininger's paper."""

    return sigma_i(M, 0) + sigma_i(M, 1)


def test_neininger():

    Ms = [markov_chain(2) for _ in range(100)]
    neins = [sigma(M) for M in Ms]
    ours = [(h_2(M) - entropy(M) ** 2) * entropy(M) ** 3 for M in Ms]

    cell_text = []

    for i in range(len(Ms)):
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

   Ms = [markov_chain(i) for i in range(2, 10)]
   ents = [entropy(M) for M in Ms]

   for i in range(len(Ms)):
     print("The Markov chain:")
     input(Ms[i])
     print("Its entropy:")
     input(ents[i])


def psi(n):
    """Returns the Psi vector: a 1-D array of size n filled with ones.

    Args:
        n (int): Size of the Markov chain.

    Returns:
        (int array): The Psi vector
    """

    return np.ones(n)

def fast_word_generator(M,f,n,N):
    'Outputs N words of size n from M'
    probas = np.random.rand(N,n)
    m00=M[0,0]
    m10=M[1,0]
    d = {0:m00, 1:m10}
    words = [[ f[ int(probas[i,0]>m00) ] ] for i in range(N)]

    for j in range(1,n):
        for i in range(N):
            w = words[i][-1]
            words[i].append( f[ (w + int(probas[i,j] > d[w])) % 2 ] )

    return words

def markov_source(M, f, n):
  """Outputs a word of size n from a Markov source (M, f)
  !! Now only works with chains of size 2 !!

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

    #next_state = 0
    proba_stack = M[current_state, 0]

    current_state = int(probas[i] > proba_stack)
    #while probas[i] > proba_stack:

    #  next_state += 1
    #  proba_stack += M[current_state, next_state]

    #current_state = next_state
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
from math import sqrt
import seaborn as sns
#import pandas as pd
from scipy import stats
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))




def redundancy_histograms(random_markov=False):
    """Makes a simulation of the redundancy distribution for different word length values,
    and prints the corresponding histograms.
    """
    style_mode = False
    fast_mode = input("Do you want fast mode activated? Y/n/s (default = true, s = style_mode)")
    if fast_mode == 'n':
      fast_mode = False
    elif fast_mode == 's':
      style_mode = True
    else:
      fast_mode = True

    i = 1000
    length_values = [10*i, 50*i, 100*i]

    if not random_markov:
        p_a = 0.9
        M = np.matrix([[p_a, 1-p_a], [1-p_a, p_a]])
        h = - p_a * log (p_a, 2) - (1-p_a) * log (1-p_a, 2)
        f = [0, 1]

    else:
        N = 2
        M = markov_chain(N)
        h = entropy(M)
        f = [0, 1]

    if not fast_mode:
        print("\nUsing a random Markov chain of size " + str(N))
        input(M)
        print("\nIts state function f is:")
        input(f)
        print("\nIts entropy is:")
        input(h)
        print("\nIts h2 is:")
        input(h_2(M))
        print("\nh2-h^2 is:")
        input(h_2(M)-h**2)

    exps = [dict() for _ in range(3)]


    for i, n_word in enumerate(length_values):
        exp = exps[i]

        if fast_mode:
            n = n_word
            n_exp = 200

        elif style_mode:
            n = int(n_word / 100)
            n_exp = 100

        else:
            try:
                n = int(input("\nChoose size of words to test (default %d)" % n_word))
            except:
                n = n_word

            try:
                n_exp = int(input("\nHow many experiments do I run? (default %d)" % 200))
            except:
                n_exp = 200

        print("\nNow testing words of size %d, doing %d experiments" % (n, n_exp))
        exp['n_exp'] = n_exp
        exp['n_word'] = n

        # Runs LZ78 over n_exp samples of words of length n from
        # the Markov chain M
        word_gen = word_generator(M, f, n)
        l =[word_gen() for _ in range(n_exp)]
        #l = fast_word_generator(M,f,n,n_exp)
        c = [compress(w) for w in l]
        m = [len(x) for x in c]
        exp['data'] = m





        # Theoretical mean
        mean = h * n / log(n,2)
        if not fast_mode:
            input("\nTheoretical mean is {}".format(mean))
        exp['mean'] = mean


        # Theoretical variances
        std_nein = sqrt (sigma(M) * n) / log(n, 2)
        std_szpan = sqrt ( -(h_2(M) - (h**2)) * h**3 * n )
        exp['std_nein'] = std_nein
        exp['std_szpan'] = std_szpan

        if not fast_mode:
            input("\nComputed std_szpan and std_nein are: {}, {}".format(std_nein, std_szpan))

        # Normalized values from Nein and Szpan papers
        d_Nein = [(m_n - mean) / std_nein  for m_n in m]
        d_Szpan = [(m_n - mean) / std_szpan for m_n in m]
        exp['d_nein'] = d_Nein
        exp['d_szpan'] = d_Szpan


        # Empirical variance and mean
        # Samples corresponding to normal distribution p
        mu, std = norm.fit(m)
        exp['mu'] = mu
        exp['std'] = std

        if not fast_mode:
            input("\nFitting mean and variance are mu=%f and var=%f" % (mu, std**2))

        # Empirical variance, theoretical mean
        # also theoretical mean and theoretical variance
        xmins = [(min(m)-me)/v for me in [mu, mean] for v in [std, std_nein, std_szpan]]
        xmaxs = [(max(m)-me)/v for me in [mu, mean] for v in [std, std_nein, std_szpan]]

        # 0 = empirical mean and variance
        # 1 = mu and std_nein
        # 2 = mu and std_szpan
        # 3 = mean and std
        # 4 = mean and std_nein
        # 5 = mean and std_szpan
        for i in range(len(xmins)):
            xmin, xmax = xmins[i], xmaxs[i]
            x = np.linspace(xmin, xmax)
            p = stats.norm(0,1).pdf(x)
            exp['x' + str(i)] = x
            exp['p' + str(i)] = p


    # Plotting raw M_n values
    figs, axs = plt.subplots(1, len(length_values), tight_layout=True)
    figs.suptitle('M_n values histogram')

    for i in [0, 1, 2]:
        sns.distplot(exps[i]['data'], ax=axs[i], rug=True, kde=False)
        axs[i].set_title('n_word = ' + str(exps[i]['n_word']) + ', n_exp = ' + str(exps[i]['n_exp']))
        axs[i].set_xlabel('M_n') ; axs[i].set_ylabel('Counts')

    # k = 0 is empirical mean and variance
    # k = 1 is theoretical mean and empirical variance
    figs_mean_std, axs_mean_std = plt.subplots(1, len(length_values), tight_layout=True)
    figs_mean_std.suptitle('M_n distribution normalized with theoretical mean and empirical variance')
    figs_normalized, axs_normalized = plt.subplots(1, len(length_values), tight_layout=True)
    figs_normalized.suptitle('Normalized M_n using empirical mean and variance')

    axes1 = [axs_mean_std, axs_normalized]
    means = ['mu', 'mean']

    for k in [0,1]:
        ax = axes1[k]
        mean = means[k]

        for i in [0, 1, 2]:
            sns.distplot((exps[i]['data']-exps[i][mean])/exps[i]['std'], ax=ax[i], rug=True)
            ax[i].set_title('n_word = ' + str(exps[i]['n_word']) + ', n_exp = ' + str(exps[i]['n_exp']))
            ax[i].set_xlabel('$(M_n -' + mean +' / \sigma$')
            ax[i].set_ylabel('Frequency')

            ax[i].plot(exps[i]['x'+str(k*3)], exps[i]['p'+str(k*3)], color='red')

    # k=0 is plotting distributions centered with empirical mean (mu and nein/szpan)
    # k=1 is plotting distributions norm with theoretical mean and variances (mu and nein/szpan)
    figs1, axs1 = plt.subplots(2, len(length_values), tight_layout=True)
    figs1.suptitle('M_n distribution normalized with empirical mean and theoretical variances')
    figs2, axs2 = plt.subplots(2, len(length_values), tight_layout=True)
    figs2.suptitle('M_n distribution norm with theoretical mean and variances')

    axes = [axs1, axs2]
    means = ['mu', 'mean']

    for k in [0,1]:

        ax = axes[k]
        mean = means[k]

        # three subplots
        for i in [0, 1, 2]:

            for (l, std) in enumerate(['std_nein', 'std_szpan']):

                sns.distplot((exps[i]['data']-exps[i][mean])/exps[i][std] , ax=ax[l][i], rug=True)
                ax[l][i].set_title('n_word = ' + str(exps[i]['n_word']) + ', n_exp = ' + str(exps[i]['n_exp']))
                ax[l][i].set_xlabel('$(M_n-'+ mean +') / '+std+'$')
                ax[l][i].set_ylabel('Frequency')

                if k == 1:
                    k+=1
                ax[l][i].plot(exps[i]['x'+str(k+l+1)], exps[i]['p'+str(k+l+1)], color='red')


    top=0.7
    figs1.subplots_adjust(top=top)
    figs_normalized.subplots_adjust(top=top)
    figs2.subplots_adjust(top=top)
    figs_mean_std.subplots_adjust(top=top)
    print("Done")
    plt.show()



if __name__ == "__main__":
  redundancy_histograms(random_markov=True)
