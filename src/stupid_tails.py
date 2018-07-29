"""
Stupid implementation of tails.py
Let's code !
"""

from markov import markov_chain, markov_iter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Instruction_Unclear(Exception):
    pass

def run_experiment(n, M=None):

    exp = dict()

    if M is None:
        M = markov_chain(2)

    exp["M"] = M
    exp["n"] = n

    parse = set("")

    tail_symbols = ""

    for k in range(1, n + 1):
        seq = ""
        m_iter = markov_iter(M)

        while seq in parse:
            seq += next(m_iter)

        parse.add(seq)
        tail_symbols += next(m_iter)

        # print("\tThis tree was {} characters long,\n\twith tail symbol {}".format(len(seq), tail_symbols[-1]))

        exp[str(k)] = seq

    exp["tail_symbols"] = tail_symbols

    return exp


def run_simulation(n_exp, n, c, M=None):

    exps = []

    if M is None:
        M = markov_chain(2)


    print("\nGenerating trees with {} nodes.".format(n))

    for _ in range(n_exp):

        exp = run_experiment(n, M)
        exps.append(exp)

    return exps

    
def get_summary(exps):

    summary = dict()
    summary["M"] = M
    summary["n_exp"] = n_exp
    summary["n_range"] = [n]
    summary["exps"] = [exps]

    variables = compute_variables(exps, n_exp, n, c)
    summary["variables"] = variables

    print("Average total path length: {}".format(variables["mean_lnc"]))

    return (exps, summary)


def compute_variables(exps, n_exp, n, c):
    """ Computes $E[T_n^c]$, etc. """

    lnc = lambda exp: sum(len(exp[str(k)]) for k in range(1, n + 1))
    tnc = lambda exp: exp["tail_symbols"].count(c)

    mean_tnc = sum(tnc(exp) for exp in exps) / n_exp
    mean_lnc = sum(lnc(exp) for exp in exps) / n_exp

    mean_tnclnc = sum(tnc(exp) * lnc(exp) for exp in exps) / n_exp

    cov_tnclnc = mean_tnclnc - mean_tnc * mean_lnc

    snd_order_tnc = sum(tnc(exp) ** 2 for exp in exps) / n_exp
    snd_order_lnc = sum(lnc(exp) ** 2 for exp in exps) / n_exp

    var_tnc = snd_order_tnc - (mean_tnc) ** 2
    var_lnc = snd_order_lnc - (mean_lnc) ** 2

    d = dict()

    for name in [
        "mean_tnc",
        "mean_lnc",
        "mean_tnclnc",
        "cov_tnclnc",
        "snd_order_tnc",
        "snd_order_lnc",
        "var_tnc",
        "var_lnc",
    ]:
        d[name] = eval(name)

    return d


def print_summary(summary):
    print(
        "\nThis is a set of {} experiments on the range {}".format(
            summary["n_exp"], summary["n_range"]
        )
    )
    print("The results are :")
    print(
        *["{} = {}".format(name, var) for (name, var) in summary["variables"].items()],
        sep="\n"
    )


def run_range_simulation(n_exp, ns, c):
    """ Run simulation over a range of values of n """

    sims = []
    M = markov_chain(2)

    from progress.bar import Bar

    bar = Bar("Progress:", max=len(ns))

    for n in ns:

        sims.append(get_summary(run_simulation(n_exp, n, c, M)))
        bar.next()

    return sims


def double_plot(fignumber, sims, ns, var1, var2, ylabel1, ylabel2):
    def extract_sum(name):
        return [summary["variables"][name] for (_, summary) in sims]

    plot1 = np.array(extract_sum(var1))
    plot2 = np.array(extract_sum(var2))


    figs, axs = plt.subplots(1, 2, num = fignumber)


    sns.regplot(ns, plot1, ax=axs[0])
    axs[0].set_ylabel(ylabel1)
    sns.regplot(ns, plot2, ax=axs[1])
    axs[1].set_ylabel(ylabel2)

    for ax in axs:
        ax.set_xlabel("Number of sequences n")


def simple_plot(fignumber, sims, ns, var, ylabel):
    def extract_sum(name):
        if name == "cov_tnclnc":
            return [summary["variables"][name] for (_, summary) in sims]
        else:
            return [summary["variables"][name] for (_, summary) in sims]

    plot = extract_sum(var)

    plt.figure(num = fignumber)
    ax = plt.subplot()
    sns.regplot(ns, plot, ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Number of sequences n")


def main():
    import sys

    if sys.argv[1] == '-s':
        ns = list(range(10, 10000 + 10, 10))
        n_exp = 700

        sims = run_range_simulation(n_exp, ns, "0")

        np.save(sys.argv[2], (sims, ns))

    elif sys.argv[1] == '-load':
        sims, ns = np.load(sys.argv[2])

    else:
        raise Instruction_Unclear


    ns = list(ns)
    print(ns)

    print_summary(sims[0][1])

    import matplotlib.pyplot as plt
    import seaborn as sns

    double_plot(1, sims, ns, "var_tnc", "var_lnc", r"$Var({T_n}^c)$", r"$Var({L_n}^c)$")
    simple_plot(3, sims, ns, "cov_tnclnc", r"$Cov({T_n}^c, {L_n}^c)$")
    double_plot(2, sims, ns, "mean_tnc", "mean_lnc", r"${T_n}^c$", r"${L_n}^c$")
    plt.show()
    # Watching mean_tnc and mean_lnc


if __name__ == "__main__":
    main()
