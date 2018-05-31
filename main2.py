import numpy as np # learn more: https://python.org/pypi/np
from math import log
import scipy
from scipy.stats import norm
import sys

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




def simulation(random_markov=False, filesave="experiment_data.npy", length_values=None, n_exp=None):
    """Makes a simulation of the redundancy distribution for different word length values.
    """
    i = 1000
    if length_values is None:
        length_values = [10*i, 50*i, 100*i]

        style_mode = False

        fast_mode = input("Do you want fast mode activated? Y/n/s (default = true, s = style_mode)")

        if fast_mode == 'n':
          fast_mode = False

        elif fast_mode == 's':
            style_mode = True
            length_values = [500, 1000, 2000]
            n_exp = 100

        else:
            fast_mode = True
            print("\nChoose words lengths:\n")
            length_values = [int(input(str(i) + ": ")) for i in range(3)]
            n_exp = int(input("Number of experiments"))

    else:
        if n_exp is None:
            n_exp = int(input("Specifiy number of experiments"))

        fast_mode = True
        style_mode = False


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

    exps = [dict() for _ in range(len(length_values))]


    for i, n in enumerate(length_values):
        exp = exps[i]

        exp['h'] = h
        exp['M'] = M

        if not (fast_mode or style_mode):
            try:
                n = int(input("\nChoose size of words to test (default %d)" % n))
            except:
                n = n

            try:
                n = int(input("\nHow many experiments do I run? (default %d)" % 200))
            except:
                n = 200

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

    input("\nNow savings experiments to " + filesave)
    np.save(filesave, exps)

    return exps, fast_mode


def data_analysis(exps=None, random_markov=True, datafile=None, filesave="experiment_data.npy", length_values=None, n_exp=None, fast_mode=False):

    if exps is None:
        if datafile is None:
            exps, fast_mode = simulation(random_markov=random_markov, filesave=filesave,
                length_values=length_values, n_exp=n_exp)

        else:
            fast_mode = False
            exps = np.load(datafile)

    for exp in exps:
        n = exp['n_word']
        n_exp = exp['n_exp']
        h = exp['h']
        M = exp['M']

        print("\n ===== This experiment has %d samples of words of size %d =====" % (n_exp, n))

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
        d_Nein = [(m_n - mean) / std_nein  for m_n in exp['data']]
        d_Szpan = [(m_n - mean) / std_szpan for m_n in exp['data']]
        exp['d_nein'] = d_Nein
        exp['d_szpan'] = d_Szpan


        # Empirical variance and mean
        # Samples corresponding to normal distribution p
        mu, std = norm.fit(exp['data'])
        exp['mu'] = mu
        exp['std'] = std

        if not fast_mode:
            input("\nFitting mean and variance are mu=%f and var=%f" % (mu, std**2))

        # Empirical variance, theoretical mean
        # also theoretical mean and theoretical variance
        mi = min(exp['data'])
        ma = max(exp['data'])
        xmins = [(mi-me)/v for me in [mu, mean] for v in [std, std_nein, std_szpan]]
        xmaxs = [(ma-me)/v for me in [mu, mean] for v in [std, std_nein, std_szpan]]

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


    if datafile is not None:
        input("\nNow savings data analysis to " + datafile)
        np.save(datafile, exps)

    return exps


def print_histograms(random_markov=True, datafile=None, fast_mode=True):
    """Prints the histograms corresponding to datasets of words generated by a
    Markov source."""
    # Plotting raw M_n values
    exps = data_analysis(random_markov=random_markov, datafile=datafile, fast_mode=fast_mode)

    figs_raw, axs = plt.subplots(1, len(exps))

    for i in [0, 1, 2]:
        sns.distplot(exps[i]['data'], ax=axs[i], rug=True, kde=False, bins='auto')
        axs[i].set_title('$n_{word} = ' + str(exps[i]['n_word']) + ', n_{exp} = ' + str(exps[i]['n_exp'])+'$')
        axs[i].set_xlabel('M_n') ; axs[i].set_ylabel('Counts')

    # k = 0 is empirical mean and variance
    # k = 1 is theoretical mean and empirical variance
    figs_mean_std, axs_mean_std = plt.subplots(1, len(exps))
    figs_normalized, axs_normalized = plt.subplots(1, len(exps))

    axs = [axs_normalized, axs_mean_std]
    means = ['mu', 'mean']

    for k in [0,1]:
        ax = axs[k]
        mean = means[k]

        for i in [0, 1, 2]:
            sns.distplot((exps[i]['data']-exps[i][mean])/exps[i]['std'], ax=ax[i], rug=True,
                        label=r'Simulation $\frac{M_n -' + ("\mu" if mean == 'mu' else 'E_{theor}') +'}{\sigma}$')
            ax[i].set_title('$n_{word} = ' + str(exps[i]['n_word']) + ', n_{exp} = ' + str(exps[i]['n_exp']) + '$')
            ax[i].set_ylabel('Frequency')

            ax[i].plot(exps[i]['x'+str(k*3)], exps[i]['p'+str(k*3)], color='red', label='$\mathcal{N}(0,1)$')

    # k=0 is plotting distributions centered with empirical mean (mu and nein/szpan)
    # k=1 is plotting distributions norm with theoretical mean and variances (mu and nein/szpan)
    figs1, axs1 = plt.subplots(1, len(exps))
    figs2, axs2 = plt.subplots(1, len(exps))
    figs4, axs4 = plt.subplots(1, len(exps))
    figs5, axs5 = plt.subplots(1, len(exps))

    figs = [figs1, figs2, figs4, figs5]
    axes = [axs1, axs2, axs4, axs5]
    means = ['mu', 'mu', 'mean', 'mean']
    stds = ['std_nein', 'std_szpan'] * 2

    for k in [0 ,1 , 2, 3]:

        ax = axes[k]
        mean = means[k]
        std = stds[k]

        # three subplots
        for i in [0, 1, 2]:

            sns.distplot((exps[i]['data']-exps[i][mean])/exps[i][std] , ax=ax[i], rug=True)
            ax[i].set_title('$n_{word} = ' + str(exps[i]['n_word']) + ', n_{exp} = ' + str(exps[i]['n_exp']) + '$')
            ax[i].set_xlabel('$(M_n-'+ ('\mu' if mean=='mu' else 'E_{theor}') +') / '+std+'$')
            ax[i].set_ylabel('Frequency')

            s=0
            if k>= 2:
                s=1
            ax[i].plot(exps[i]['x'+str(k+1+s)], exps[i]['p'+str(k+1+s)], color='red', label='$\mathcal{N}(0,1)$')



    figs_raw.suptitle('Histogram of the values of M_n')
    figs_mean_std.suptitle('$M_n$ distribution, normalized with theoretical mean and empirical variance')
    figs_normalized.suptitle('$M_n$ distribution, normalized using empirical mean and variance')

    i = 0
    for n1 in ['empirical', 'theoretical']:
        for n2 in ['Neininger', 'Szpankowski']:
            figs[i].suptitle('$M_n$ distribution, normalized with {} mean and {} variance.'.format(n1,n2))
            i += 1

    top=0.90
    bottom = 0.04
    left=0.08
    right=0.98
    hspace=0.28
    wspace=0.25

    for ax in axs_normalized:
        ax.legend()

    for ax in axs_mean_std:
        ax.legend()

    for ax in axes:
        for a in ax:
            a.legend()

    for f in figs:
        f.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

    figs_normalized.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    figs_mean_std.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    figs_raw.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

    print("Done")
    plt.show()



def data_loading(filesave=None, datafile=None):
    """Loads experiments by either simulating them or loading them from memory.
    Analyses experiments that previously weren't
    """

    n_exp = 500
    N = 2
    M = markov_chain(N)
    h = entropy(M)
    f = [0, 1]

    ns = list(range(100, 20000, 200))

    if datafile is None:
        exps, _ = data_analysis(filesave=filesave, length_values = ns, n_exp=500)

    else:
        exps = np.load(datafile)

    try:
        mu = exps[0]['mu']
    except:
        print("Analysing the experiments")
        fast_mode = False if input("Do you want to see the values? y/N (defaut is false)") == 'Y' else True
        exps = data_analysis(exps=exps, datafile=datafile, fast_mode=fast_mode)

    return exps, ns

def analysing_theoretical_std(filesave=None, datafile=None):
    """Computes graphs of theoretical standard deviation versus empirical ones.
    """
    exps, ns = data_loading(filesave=filesave, datafile=datafile)

    stds = [exp['std'] for exp in exps]
    variances = [exp['std'] ** 2 for exp in exps]
    neins = [exp['std_nein'] for exp in exps]
    diff = [stds[i]-neins[i] for i in range(len(stds))]

    figs, axs = plt.subplots(1, 2, tight_layout=True)

    axs[0].plot(ns, stds, label="$\sigma$")
    axs[0].plot(ns, neins, label=r'$\sigma_{Neininger}$')
    axs[0].set_title(r'Empirical standard deviation ($\sigma$) and theoretical ($\sigma_{Neininger}$)')

    axs[1].plot(ns, diff, label=r'$\Delta \sigma = \sigma - \sigma_{Neininger}$')
    axs[1].set_title('Difference between standard deviations')

    for ax in axs:
        ax.set_xlabel("Word length n")
        ax.legend()

def analysing_theoretical_mean(filesave=None, datafile=None):
    """Computes graphs of theoretical means versus empirical ones.
    The goal is to identify a \log n tendency"""
    exps, ns = data_loading(filesave=filesave, datafile=datafile)

    mus = [exp['mu'] for exp in exps]
    means = [exp['mean'] for exp in exps]
    diff = [mus[i]-means[i] for i in range(len(mus))]
    inv_diff = [1/d for d in diff]
    n_log2 = [n / log(n, 2)**2 for n in ns]
    n_log = [n / log(n, 2) for n in ns]
    n_log32 = [n / log(n, 2)**(1.34) for n in ns]
    logs = [log(n,2) for n in ns]
    div = [mus[i]/means[i] for i in range(len(mus))]
    inv = [1/n for n in ns]
    inv_logs = [1/x for x in logs]
    es = [0.01*i for i in range(2,8)]
    different_logs = [[n / log(n, 2) ** (1.30+e) for n in ns] for e in es]
    asympt = [diff[i] * sqrt(ns[i]) / log(ns[i], 2) for i in range(len(ns))]

    figs, axs = plt.subplots(1, 3, tight_layout=True)

    axs[0].plot(ns, mus, color="orange", label="$\mu$")
    axs[0].plot(ns, means, color="green", label=r'$E_{th} = \frac{nh}{\log_2(n)}$', linestyle="-")
    axs[0].set_title("Empirical ($\mu$) and theoretical mean ($E_{th}$) plots")

    axs[1].plot(ns, n_log2, color="green", label=r'$\frac{n}{\log_2^2 n}$')

    for (i,e) in enumerate(es):
        axs[1].plot(ns, different_logs[i], label=r'$\frac{n}{(\log_2(n))^{' + str(1.30+e) + '}}$')

    axs[1].plot(ns, diff, color="black", label="$\Delta E$", linestyle="-")

    axs[1].plot(ns, n_log, color="red", label=r'$\frac{n}{\log_2 n}$')
    axs[1].set_title("Difference $\Delta E = \mu-E_{th}, and approximations$")

    # axs[2].plot(ns, div, color='blue', label="$(\Delta E)^{-1}$", linestyle="-")
    # axs[2].plot(ns, logs, color="green", label="$\log_2 n$")
    # axs[2].plot(ns, inv_logs, color="red", label=r'$\frac{1}{\log_2 n}$')
    # axs[2].set_title("Division $\mu/mean$")

    axs[2].plot(ns, div, color='blue', label=r'$\frac{\sqrt{n}(\mu-E_{th})}{\log_2 n}$', linestyle="-")
    axs[2].set_title(r'Verifying $\frac{\sqrt{n}(\mu-E_{th})}{\log_2 n} = o(1)$')

    # axs[2].plot(ns, inv_diff, color='blue', label="$(\Delta E)^{-1}", linestyle="-")
    # axs[2].set_title("Inverse of $\Delta E$ with logarithmic n")
    # axs[2].set_xscale('log')

    axs[0].set_ylabel("Different computations for number of phrases expectancy E(M_n)")
    for ax in axs:
            ax.set_xlabel("Word length n")
            ax.legend()

    plt.show()


def files_choice(arg, name):
    """Return None, None if arguments weren't provided at program launch"""
    try:
        if arg == '--file':
            return name, None
        elif arg == '--save':
            return None, name
        else:
            return None, None
    except:
        return None, None


if __name__ == "__main__":

    datafile, filesave = files_choice(sys.argv[2], sys.argv[3])

    if len(sys.argv) > 1:

        if sys.argv[1] == '-s':
            simulation(random_markov=True, filesave = sys.argv[2])

        if 'm' in sys.argv[1]:
            analysing_theoretical_mean(datafile=datafile, filesave=filesave)

        if 'v' in sys.argv[1]:
            analysing_theoretical_std(datafile=datafile, filesave=filesave)

        else:
            print_histograms(random_markov=True, datafile = sys.argv[1])

        plt.show()

    else:
        print_histograms(random_markov=True)
