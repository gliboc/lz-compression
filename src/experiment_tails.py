"""Functions to treat data from experiments"""
import numpy as np
from markov import markov_source2
from lempelziv import compress2
from progress.bar import Bar
from glob import glob
from tails import get_summary

def show_single_sims(filename):

    exps, summ = np.load(filename)

    input("Data file {} contains {} experiments with chain {}".format(filename, len(exps), exps[0]["M"]))
    print("These are their lengths:")
    print(*["(n_exp, n_word) = ({}, {})".format(exp["n_exp"], exp["n_word"]) for exp in exps], sep="\t\n")


def show_all_sims(dirname):

    names = glob(dirname + '*.npy')
    print(names)

    for filename in names:
        show_single_sims(filename)


def union(sims1, sims2):
    """Unites the experiments if they were done on the same range of
    values ns and the same Markov source M
    """

    ns1 = [exps[0]["n"] for (exps, _) in sims1]
    ns2 = [exps[0]["n"] for (exps, _) in sims2]
    M1 = sims1[0][1]["M"]
    M2 = sims2[0][1]["M"]

    if not (M1 == M2).all() or ns1 != ns2:
        print("Error : different experimental conditions")
        exit()

    sims = []
    for ((exps1, _), (exps2, _)) in zip(sims1, sims2):
        exps = exps1 + exps2
        exps[0]["n_exp"] = len(exps)
        
        sims.append(get_summary(exps))

    return sims


def add_all(filename):
    from parallel_tails import parallel_simu

    sims1 = np.load(filename)

    exps, summ = sims1[0]
    M = summ["M"]
    ns = [exps[0]["n"] for (exps, _) in sims1]
    n_exp = len(exps)

    print("This is a set of {} experiments on range {}".format(n_exp, ns))

    n_new = int(input("How many experiments do you want to add ? "))

    sims2 = parallel_simu(M, ns, n_new)

    sims = union(sims1, sims2)

    print("Now this set has {} experiments".format(len(sims[0][0])))

    np.save(filename, sims)


if __name__ == "__main__":

    import sys

    if sys.argv[1] == '-add':
        add_all(sys.argv[2])

    elif sys.argv[1] == '-show':
        show_all_sims(sys.argv[2])

    elif sys.argv[1] == '-single':
        show_single_sims(sys.argv[2])
