"""Functions to treat data from experiments"""
import numpy as np
from markov import markov_source2
from lempelziv import compress2
from progress.bar import Bar


def add_words(filename):
    """Loads experimental data and generates new words"""

    exps = np.load(filename)

    for exp in exps:
        M = exp["M"]
        n_exp = exp["n_exp"]
        n_word = exp["n_word"]
        print("{} words of size {} generated using Markov chain {}".format(n_exp, n_word, M))
        assert(len(exp["data"]) == n_exp)

        n_add = int(input("How many words do you want to add to this experiment ?"))

        bar = Bar("Progress", max=n_add)
        new_words = []
        for _ in range(n_add):
            new_words.append(compress2(markov_source2(M, n_word)))
            bar.next()

        exp["data"] += new_words
        exp["n_exp"] += n_add

        print("There are now {} words in this experiment".format(len(exp["data"])))
    
    input("\nNow savings experiments to " + filename)
    np.save(filename, exps)        


def add_all(filename, n_add):
    """Loads experimental data and generates new words"""

    exps = np.load(filename)

    for exp in exps:
        M = exp["M"]
        n_exp = exp["n_exp"]
        n_word = exp["n_word"]
        print("{} words of size {} generated using Markov chain {}".format(n_exp, n_word, M))
        assert(len(exp["data"]) == n_exp)

        print("Adding {} words".format(n_add))

        bar = Bar("Progress", max=n_add)
        new_words = []
        for _ in range(n_add):
            new_words.append(compress2(markov_source2(M, n_word)))
            bar.next()

        exp["data"] += new_words
        exp["n_exp"] += n_add

        print("There are now {} words in this experiment".format(len(exp["data"])))
    
    input("\nNow savings experiments to " + filename)
    np.save(filename, exps)        

from glob import glob


def show_single(filename):

    exps = np.load(filename)
    input("Data file {} contains {} experiments with chain {}".format(filename, len(exps), exps[0]["M"]))
    print("These are their lengths:")
    print(*["(n_exp, n_word) = ({}, {})".format(exp["n_exp"], exp["n_word"]) for exp in exps], sep="\t\n")


def show_experiments(dirname):
    
    names = glob('*.npy')

    for filename in names:
        show_single(filename)


if __name__ == "__main__":

    import sys

    if sys.argv[1] == '-all':
        n_add = int(input("How many words do you want to add to all experiments ?"))
        add_all(sys.argv[2], n_add)

    elif sys.argv[1] == '-show':
        show_experiments(sys.argv[2])

    elif sys.argv[1] == '-single':
        show_single(sys.argv[2])

    else:
        add_words(sys.argv[1])

    