"""Plotting graphs and histogram_chains using data from LZ78 applied to Markov sources

Simulation

    $ python main.py <datafile>

        Plots the usual histogram_chains from the set of experiments in <datafile>.

    $ python main.py -s <filesave>

        Runs a simulation after prompting for three coupless (n_word, n_exp)
        and saves it as <filesave>.

    $ python main.py -range <filesave>

        Runs a simulation after prompting for a range of values ns
        and saves it as <filesave>.

    $ python main.py -m --file <datafile>

        Loads the set of experiments <datafile> and plots the graphs related
        to mean analysis.

    $ python main.py -m --save <savefile>

        Runs the analysis and then saves the data - with analysis - into
        the file <savefile>.

    $ python main.py -v  (--file <datafile> | --save <savefile>)

        Works the same as the -m argument previously seen, except that it
        does variance analysis and plots.

    $ python main.py -cdf  (--file <datafile> | --save <savefile>)

        Works the same as the -m argument previously seen, except that it
        does cumulative distribution function analysis and plots.


"""

import numpy as np  # learn more: https://python.org/pypi/np
from math import log
from scipy.stats import norm

# from scipy.stats import normaltest
import sys
from szpan import var, h_2, entropy
from markov import markov_chain, markov_source2
from neininger import H, sigma2_H3
from lempelziv import compress, compress2

import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

# import pandas as pd
from scipy import stats
from eigenvalues import lambda_2, eigenvalue_std

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

DEBUG = True


def test_neininger():
    m_chains = [markov_chain(2) for _ in range(100)]
    neins = [sigma2_H3(M) for M in m_chains]
    ours = [(h_2(M) - entropy(M) ** 2) * entropy(M) ** 3 for M in m_chains]

    cell_text = []

    for i in range(len(m_chains)):
        cell_text.append(["%1.5f" % x for x in [neins[i], ours[i]]])

    columns = ["Neiningers H^3 * sigma^2", "h2 - h^2"]
    rows = [str((M[0, 0], M[1, 0])) for M in m_chains]

    plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns)

    plt.plot(neins, color="r", label="Neininger")
    plt.plot(ours, color="g", label="Szpankowski")

    plt.legend()
    plt.title("Szpan vs Nein comparison")
    plt.show()


def test_entropy():
    m_chains = [markov_chain(i) for i in range(2, 10)]
    ents = [entropy(M) for M in m_chains]

    for i in range(len(m_chains)):
        print("The Markov chain:")
        input(m_chains[i])
        print("Its entropy:")
        input(ents[i])


def fast_word_generator(M, f, n, N):
    "Outputs N words of size n from M"
    probas = np.random.rand(N, n)
    m00 = M[0, 0]
    m10 = M[1, 0]
    d = {0: m00, 1: m10}
    words = [[f[int(probas[i, 0] > m00)]] for i in range(N)]

    for j in range(1, n):
        for i in range(N):
            w = words[i][-1]
            words[i].append(f[(w + int(probas[i, j] > d[w])) % 2])

    input("Generated words")
    return words


def simulation(
    random_markov=True, filesave="experiment_data.npy", length_values=None, n_exp=None
):
    """Makes a simulation of the redundancy distribution for different word length values.
    """
    i = 1000
    if length_values is None:
        length_values = [10 * i, 50 * i, 100 * i]

        style_mode = False

        fast_mode = input(
            "Do you want fast mode activated? Y/n/s (default = true, s = style_mode) "
        )

        if fast_mode == "n":
            fast_mode = False

        elif fast_mode == "s":
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

    print("Random markov has values", random_markov)

    if not random_markov:
        p_a = 0.9
        M = np.matrix([[p_a, 1 - p_a], [1 - p_a, p_a]])
        h = -p_a * log(p_a, 2) - (1 - p_a) * log(1 - p_a, 2)
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
        input(h_2(M) - h ** 2)

    exps = [dict() for _ in range(len(length_values))]

    for i, n in enumerate(length_values):
        exp = exps[i]

        exp["h"] = h
        exp["M"] = M

        if not (fast_mode or style_mode):
            try:
                n = int(input("\nChoose size of words to test (default %d) " % n))
            except:
                n = n

            try:
                n_exp = int(
                    input("\nHow many experiments do I run? (default %d) " % 200)
                )
            except:
                n_exp = 200

        print("\nNow simulating words of size %d, doing %d experiments " % (n, n_exp))
        exp["n_exp"] = n_exp
        exp["n_word"] = n

        # Runs LZ78 over n_exp samples of words of length n from
        # the Markov chain M
        # word_gen = word_generator(M, f, n)
        # l =[word_gen() for _ in range(n_exp)]
        # l = fast_word_generator(M,f,n,n_exp)
        # c = [compress2(w) for w in l]
        # m = [len(x) for x in c]
        m = [compress2(markov_source2(M, n)) for _ in range(n_exp)]
        exp["data"] = m

    exps[0]["ns"] = length_values
    input("\nNow savings experiments to " + filesave)
    np.save(filesave, exps)

    return exps, fast_mode


def data_analysis(
    exps=None,
    random_markov=True,
    datafile=None,
    filesave="experiment_data.npy",
    length_values=None,
    n_exp=None,
    fast_mode=False,
):
    if exps is None:
        if datafile is None:
            if n_exp is None:
                n_exp = 500
            exps, fast_mode = simulation(
                random_markov=random_markov,
                filesave=filesave,
                length_values=length_values,
                n_exp=n_exp,
            )

        else:
            fast_mode = False
            exps = np.load(datafile)

    for exp in exps:
        n = exp["n_word"]
        n_exp = exp["n_exp"]
        h = exp["h"]
        M = exp["M"]

        print(
            "\n ===== This experiment has %d samples of words of size %d\nUsing chain 
            % (n_exp, n)
        )

        # Theoretical mean
        mean = h * n / log(n)
        # mean2 = H(M) * n / log(n, 2)
        H(M)  # à enlever

        if not fast_mode:
            input("\nTheoretical mean is {}".format(mean))
        exp["mean"] = mean

        # Theoretical variances
        std_nein = sqrt(sigma2_H3(M) * n) / log(n, 2)
        # std_szpan = sqrt ( -(h_2(M) - (h**2)) * h**3 * n )
        print("M is", M)
        print("n is", n)
        print("Value for variance var(M)", var(M, n))
        print("Value of log(m)", log(mean))
        #std_szpan = sqrt(abs(var(M, n)))
        std_szpan = eigenvalue_std(M, n) # computed with lambda
        std_eig = eigenvalue_std(M, n)
        exp["std_nein"] = std_nein
        exp["std_szpan"] = std_szpan

        if not fast_mode:
            input(
                "\nComputed std_szpan and std_nein are: {}, {}".format(
                    std_szpan, std_nein
                )
            )

        # Normalized values from Nein and Szpan papers
        d_nein = [(m_n - mean) / std_nein for m_n in exp["data"]]
        d_szpan = [(m_n - mean) / std_szpan for m_n in exp["data"]]
        d_eig =  [(m_n - mean) / std_eig for m_n in exp["data"]]
        exp["d_nein"] = d_nein
        exp["d_szpan"] = d_szpan
        exp["d_eig"] = d_eig

        # Empirical variance and mean
        # Samples corresponding to normal distribution p

        print("Showing the data")

        mu, std = norm.fit(exp["data"])
        exp["mu"] = mu
        exp["std"] = std

        if not fast_mode:
            input("\nFitting mean and std are mu=%f and var=%f" % (mu, std))

        # Empirical variance, theoretical mean
        # also theoretical mean and theoretical variance
        mi = min(exp["data"])
        ma = max(exp["data"])
        xmins = [(mi - me) / v for me in [mu, mean] for v in [std, std_nein, std_szpan]]
        xmaxs = [(ma - me) / v for me in [mu, mean] for v in [std, std_nein, std_szpan]]

        # 0 = empirical mean and variance
        # 1 = mu and std_nein
        # 2 = mu and std_szpan
        # 3 = mean and std
        # 4 = mean and std_nein
        # 5 = mean and std_szpan
        for i in range(len(xmins)):
            xmin, xmax = xmins[i], xmaxs[i]
            x = np.linspace(xmin, xmax)
            p = stats.norm(0, 1).pdf(x)
            exp["x" + str(i)] = x
            exp["p" + str(i)] = p

    if datafile is not None:
        input("\nNow savings data analysis to " + datafile)
        np.save(datafile, exps)

    return exps


def data_loading(filesave=None, datafile=None):
    """Loads experiments by either simulating them or loading them from memory.
    Analyses experiments that previously weren't
    """

    if datafile is None:

        print("\nChoose ns interval range(a, b, s)")

        a = int(input("a = "))
        b = int(input("b = "))
        s = int(input("s = "))
        ns = list(range(a, b, s))
        n_exp = int(input("\nHow many experiments ? "))

        exps = data_analysis(filesave=filesave, length_values=ns, n_exp=n_exp)

    else:

        print("Loading file ", datafile)

        exps = np.load(datafile)

    print("Analysing the experiments")

    fast_mode = (
        False
        if input("Do you want to see the values? y/N (defaut is false) ").lower() == "y"
        else True
    )

    exps = data_analysis(exps=exps, datafile=datafile, fast_mode=fast_mode)

    try:
        ns = exps[0]["ns"]
    except:
        ns = list(range(100, 20000, 200))  # maybe something else

    return exps, ns


def analysing_theoretical_std(filesave=None, datafile=None, save=None, save_name=None):
    """Computes graphs of theoretical standard deviation versus empirical ones.
    """
    exps, ns = data_loading(filesave=filesave, datafile=datafile)
    n_exp = exps[0]["n_exp"]

    # N = 3
    stds = [exp["std"] for exp in exps]
    # variances = [exp['std'] ** 2 for exp in exps]
    neins = [exp["std_nein"] for exp in exps]
    szpans = [exp["std_szpan"] for exp in exps]
    diff1 = [stds[i] - neins[i] for i in range(len(stds))]
    diff2 = [stds[i] - szpans[i] for i in range(len(stds))]

    # fsts = [diff[40] - log(ns[40], 2) ** (0.5 + i*0.1 + 0.3) for i in range(N)]
    try:
        fst = diff1[10] - log(ns[10], 2)
    except:
        fst = diff1[0] - log(ns[0], 2)
    # logs = [[log(n, 2) ** (0.5 + i*0.1 + 0.3) + fsts[i]  for n in ns] for i in range(N)]
    logs = [log(n, 2) + fst for n in ns]

    figs, axs = plt.subplots(1, 2, tight_layout=True)

    axs[0].plot(ns, stds, label=r"$\sigma$")
    axs[0].plot(ns, neins, label=r"$\sigma_{Neininger}$")
    axs[0].plot(ns, szpans, label=r"$\sigma_{Szpankowski}$")
    axs[0].set_title(
        r"Empirical standard deviation ($\sigma$)"
        + r" and theoretical ones ($\sigma_{Neininger}$, $\sigma_{S}$), $n_{exp}$ = "
        + str(n_exp)
    )

    axs[1].plot(ns, diff1, label=r"$\Delta \sigma = \sigma - \sigma_{Neininger}$")
    axs[1].plot(ns, diff2, label=r"$\Delta \sigma = \sigma - \sigma_{Szpan}$")

    # for i in range(N):
    #    e = 0.5 + i*0.1 + 0.3
    #    axs[1].plot(ns, logs[i], label=r'${(\log_2(n))}^{%1.2f}-%1.2f$' % (e, -fsts[i]))

    axs[1].plot(ns, logs, label=r"${(\log_2(n))}-%1.2f$" % -fst)
    axs[1].set_title(
        "Difference between standard deviations, $n_{exp}$ = " + str(n_exp)
    )

    for ax in axs:
        ax.set_xlabel("Word length n")
        ax.legend()

    if save:
        print("Saving figure as " + save_name)
        plt.savefig(save_name, dpi="figure")


def analysing_theoretical_mean(filesave=None, datafile=None, save=None, save_name=None):
    r"""Computes graphs of theoretical means versus empirical ones.
    The goal is to identify a \log n tendency"""
    exps, ns = data_loading(filesave=filesave, datafile=datafile)
    n_exp = exps[0]["n_exp"]

    input(ns)
    mus = [exp["mu"] for exp in exps]
    means = [exp["mean"] for exp in exps]
    input(means)
    diff = [mus[i] - means[i] for i in range(len(mus))]
    # inv_diff = [1/d for d in diff]
    n_log2 = [n / log(n, 2) ** 2 for n in ns]
    n_log = [n / log(n, 2) for n in ns]
    # n_log32 = [n / log(n, 2)**(1.34) for n in ns]
    # logs = [log(n,2) for n in ns]
    div = [mus[i] / means[i] for i in range(len(mus))]
    # inv = [1/n for n in ns]
    # inv_logs = [1/x for x in logs]
    es = [0.01 * i for i in range(2, 8)]
    different_logs = [[n / log(n, 2) ** (1.30 + e) for n in ns] for e in es]
    # asympt = [diff[i] * sqrt(ns[i]) / log(ns[i], 2) for i in range(len(ns))]

    figs, axs = plt.subplots(1, 3, tight_layout=True)

    axs[0].plot(ns, mus, color="orange", label=r"$\mu$")
    axs[0].plot(
        ns,
        means,
        color="green",
        label=r"$E_{\text{th}} = \frac{nh}{\log_2(n)}$",
        linestyle="-",
    )
    axs[0].set_title(r"Empirical ($\mu$) and theoretical mean ($E_{\text{th}}$) plots")

    axs[1].plot(ns, n_log2, color="green", label=r"$\frac{n}{\log_2^2 n}$")

    # for (i,e) in enumerate(es):
    #     axs[1].plot(ns, different_logs[i], label=r'$\frac{n}{(\log_2(n))^{' + str(1.30+e) + '}}$')

    axs[1].plot(ns, diff, color="black", label=r"$\Delta E$", linestyle="-")

    axs[1].plot(ns, n_log, color="red", label=r"$\frac{n}{\log_2 n}$")
    axs[1].set_title(r"Difference $\Delta E = \mu-E_{th}$, and approximations")

    # axs[2].plot(ns, div, color='blue', label="$(\Delta E)^{-1}$", linestyle="-")
    # axs[2].plot(ns, logs, color="green", label="$\log_2 n$")
    # axs[2].plot(ns, inv_logs, color="red", label=r'$\frac{1}{\log_2 n}$')
    # axs[2].set_title("Division $\mu/mean$")

    axs[2].plot(
        ns,
        div,
        color="blue",
        label=r"$\frac{\sqrt{n}(\mu-E_{th})}{\log_2 n}$",
        linestyle="-",
    )
    axs[2].set_title(r"Verifying $\frac{\sqrt{n}(\mu-E_{th})}{\log_2 n} = o(1)$")

    # axs[2].plot(ns, inv_diff, color='blue', label="$(\Delta E)^{-1}", linestyle="-")
    # axs[2].set_title("Inverse of $\Delta E$ with logarithmic n")
    # axs[2].set_xscale('log')

    axs[0].set_ylabel(
        "Different computations for number of phrases expectancy $E(M_n)$"
    )
    for ax in axs:
        ax.set_xlabel("Word length n")
        ax.legend()

    if save:
        print("Saving figure as " + save_name)
        plt.savefig(save_name, dpi="figure")


def cdf(data):
    """Outputs the cumulative distribution of a set of data"""

    ntotal = len(data)
    cdfs = []

    for x in sorted(data):
        cdfs.append(len([d for d in data if d <= x]) / ntotal)

    return cdfs


def cdf2(xs, data):

    ntotal = len(data)
    cdfs = []

    for x in xs:
        c = len([d for d in data if d <= x])
        cdfs.append(c / ntotal)

    return cdfs


def print_cdf(exp, ax, mean, std):
    """Prints the cdf associated to the experiments exps on the axes axs"""

    mu = exp[mean]
    std = exp[std]

    mi = min(exp["data"])
    ma = max(exp["data"])

    xs = np.linspace((mi - mu) / std, (ma - mu) / std, 500)
    norm_cdf = [norm.cdf(x) for x in xs]

    norm_data = (exp["data"] - mu) / std
    data_cdf = cdf2(xs, norm_data)

    ax.plot(xs, data_cdf, color="green", label="Data CDF")
    ax.plot(xs, norm_cdf, color="red", label="Normal CDF")

    ax.legend()

    ax.set_title(
        r"$n_{word} ="
        + latex_float(exp["n_word"])
        + r"\quad n_{exp} = "
        + latex_float(exp["n_exp"])
        + "$"
    )
    ax.set_xlabel(
        r"$\frac{(M_n-" + (r"\mu" if mean == "mu" else r"E_{theor}") + r")}{\sigma}$"
    )


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def print_histogram_chains(random_markov=True, datafile=None, fast_mode=True):
    """Prints the histogram_chains corresponding to datasets of words generated by a
    Markov source."""
    # Plotting raw M_n values
    exps = data_analysis(
        random_markov=random_markov, datafile=datafile, fast_mode=fast_mode
    )[-5:]
    

    # Raw histograms; not useful anymore
    # figs_raw, axs = plt.subplots(1, len(exps))
    # axs[0].set_ylabel("Counts")
    # 
    # for (i, exp) in enumerate(exps):
    #     sns.distplot(exp["data"], ax=axs[i], rug=True, kde=False, bins="auto")
    #     axs[i].set_title(
    #         r"$n_{word} ="
    #         + latex_float(exp["n_word"])
    #         + r"\quad n_{exp} = "
    #         + latex_float(exp["n_exp"])
    #         + "$"
    #     )
    #     axs[i].set_xlabel("$M_n$")

    # Normalized distirbutions plots, using mu, mean and empirical std
    # k = 0 is empirical mean and variance
    # k = 1 is theoretical mean and empirical variance
    empirical_std = False

    if empirical_std:
        figs_mean_std, axs_mean_std = plt.subplots(1, len(exps))
        figs_normalized, axs_normalized = plt.subplots(1, len(exps))

        axs = [axs_normalized, axs_mean_std]
        means = ["mu", "mean"]

        for k in [0, 1]:
            ax = axs[k]
            mean = means[k]
            ax[0].set_ylabel("Frequency")

            for (i, exp) in enumerate(exps):
                norm_distrib = (exp["data"] - exp[mean]) / exp["std"]
                # statistic, pvalue = normaltest(norm_distrib)
                sns.distplot(
                    norm_distrib,
                    ax=ax[i],
                    rug=True,
                    label=r"Simulation $\frac{M_n -"
                    + (r"\mu" if mean == "mu" else r"E_{theor}")
                    + r"}{\sigma}$",
                )
                # ax[i].text(-4, 0.3, r'$test=%1.3f$' % statistic)
                # ax[i].text(-4, 0.28, r'$pvalue=%1.3f$' % pvalue)
                ax[i].set_title(
                    r"$n_{word} ="
                    + latex_float(exp["n_word"])
                    + "\quad n_{exp} = "
                    + latex_float(exp["n_exp"])
                    + "$"
                )
                ax[i].set_xlabel(
                    r"$\frac{(M_n-"
                    + ("\mu" if mean == "mu" else "E_{theor}")
                    + ")}{\sigma}$"
                )

                # Print awaited normal distribution
                ax[i].plot(
                    exp["x" + str(k * 3)],
                    exp["p" + str(k * 3)],
                    color="red",
                    label="$\mathcal{N}(0,1)$",
                )

    # k=0 is plotting distributions centered with empirical mean (mu and nein/szpan)
    # k=1 is plotting distributions norm with theoretical mean and variances (mu and nein/szpan)
    figs1, axs1 = plt.subplots(1, len(exps))
    figs2, axs2 = plt.subplots(1, len(exps))
    figs4, axs4 = plt.subplots(1, len(exps))
    figs5, axs5 = plt.subplots(1, len(exps))

    figs = [figs1, figs2, figs4, figs5]
    axes = [axs1, axs2, axs4, axs5]
    means = ["mu", "mu", "mean", "mean"]
    stds = ["std_nein", "std_szpan"] * 2
    stds_pprint = ["std_{Nein}", "std_{Szpan}"] * 2

    for k in [0, 1, 2, 3]:

        ax = axes[k]
        mean = means[k]
        std = stds[k]
        std_pprint = stds_pprint[k]

        # three subplots
        for (i, exp) in enumerate(exps):
            norm_distrib = (exp["data"] - exp[mean]) / exp[std]
            # statistic, pvalue = normaltest(norm_distrib)
            sns.distplot(norm_distrib, ax=ax[i], rug=True)
            # ax[i].text(-4, 0.3, r'$test=%1.3f$' % statistic)
            # ax[i].text(-4, 0.28, r'$pvalue=%1.3f$' % pvalue)

            ax[i].set_title(
                r"$n_{word} ="
                + latex_float(exp["n_word"])
                + "\quad n_{exp} = "
                + latex_float(exp["n_exp"])
                + "$"
            )
            ax[i].set_xlabel(
                "$(M_n-"
                + ("\mu" if mean == "mu" else "E_{theor}")
                + ") / "
                + std_pprint
                + "$"
            )
            ax[i].set_ylabel("Frequency")

            s = 0
            if k >= 2:
                s = 1
            ax[i].plot(
                exp["x" + str(k + 1 + s)],
                exp["p" + str(k + 1 + s)],
                color="red",
                label="$\mathcal{N}(0,1)$",
            )

    # figs_raw.suptitle(
    #     "Histogram_chains of the values of $M_n$ for different word lengths"
    # )
    # figs_mean_std.suptitle(
    #     "$M_n$ distribution, normalized with theoretical mean"
    #     + "($E_{th}$) and empirical variance ($\sigma^2$)"
    # )
    # figs_normalized.suptitle(
    #     "$M_n$ distribution, normalized using empirical mean ($\mu$) and variance ($\sigma^2$)"
    # )

    i = 0
    for n1 in ["empirical", "theoretical"]:
        for n2 in ["Neininger", "Szpankowski"]:
            figs[i].suptitle(
                "$M_n$ distribution, normalized with {} mean and {} variance.".format(
                    n1, n2
                )
            )
            i += 1

    top = 0.91
    bottom = 0.06
    left = 0.04
    right = 0.99
    hspace = 0.32
    wspace = 0.13

    # for ax in axs_normalized:
    #     ax.legend()

    # for ax in axs_mean_std:
    #     ax.legend()

    for ax in axes:
        for a in ax:
            a.legend()

    for f in figs:
        f.subplots_adjust(
            top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
        )

    # figs_normalized.subplots_adjust(
    #     top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
    # )
    # figs_mean_std.subplots_adjust(
    #     top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
    # )
    # figs_raw.subplots_adjust(
    #     top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
    # )

    print("Done")
    plt.show()


def files_choice(arg, name):
    """Return None, None if arguments weren't provided at program launch"""
    try:
        if arg == "--file":
            return name, None
        elif arg == "--save":
            return None, name
        else:
            return None, None
    except:
        return None, None


if __name__ == "__main__":
    try:
        df, fs = files_choice(sys.argv[2], sys.argv[3])
    except:
        df, fs = None, None

    if len(sys.argv) > 1:
        # x = input("Do you want to save the generated figures ? y/N ")
        x = False  # temporary
        # save = False if (x.lower() == 'n') else bool(x)
        save = False

        if sys.argv[1] == "-s":

            simulation(random_markov=True, filesave=sys.argv[2])

        elif sys.argv[1] == "-m":

            save_name = None

            if save:
                save_name = input("What prefix for mean analysis figures ?")

            analysing_theoretical_mean(
                datafile=df, filesave=fs, save=save, save_name=save_name
            )

        elif sys.argv[1] == "-v":

            save_name = None

            if save:

                save_name = input("What prefix for std analysis figures ?")

            analysing_theoretical_std(
                datafile=df, filesave=fs, save=save, save_name=save_name
            )

        elif sys.argv[1] == "-cdf":

            experiments, ns = data_loading(datafile=df)
            figs, axs = plt.subplots(1, len(exps))

            for (exp_index, one_exp) in enumerate(experiments):
                print_cdf(one_exp, axs[exp_ind], "mu", "std")

            figs.suptitle("Cumulative distribution function plots for normalized $M_n$")

        elif sys.argv[1] == "-range":

            print("Prompting for range of simulation `range(a, b, s)`")
            a = int(input("a = "))
            b = int(input("b = "))
            s = int(input("s = "))
            ns = list(range(a, b, s))
            n_exp = int(input("\nHow many experiments ? "))

            simulation(
                random_markov=True, filesave=sys.argv[2], length_values=ns, n_exp=n_exp
            )

        else:
            print_histogram_chains(random_markov=True, datafile=sys.argv[1])

        plt.show()

    else:
        print_histogram_chains(random_markov=True)
