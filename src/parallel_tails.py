from multiprocessing import Process, Queue, Pool
import numpy as np
import os
from math import log
from stupid_tails import run_experiment, get_summary
from markov import markov_chain, markov_iter_rng


def rng_gen(N, n):

    formula = int(N * 3 * n * log(n, 2))
    M = min(formula, 20_000_000)

    # if formula > 100_000_000:
    #     print("Chose constant")
    # else:
    #     print("Chose formula")

    while True:
        rands = np.random.rand(M)
        # print("Generated {} random numbers".format(len(rands)))
        for r in rands:
            yield r
        print("Had to refresh")


def run_simu_mr(M, n, N):

    rng = rng_gen(N, n)
    exps = []

    for _ in range(N):
        exp = dict()
        exp["M"] = M
        exp["n"] = n

        tail_symbols = ""
        parse = set("")

        for k in range(1, n + 1):
            seq = ""
            m_iter = markov_iter_rng(M, rng)

            while seq in parse:
                seq += next(m_iter)

            parse.add(seq)
            tail_symbols += next(m_iter)

            exp[str(k)] = seq

        exp["tail_symbols"] = tail_symbols

        exps.append(exp)

    return exps


def main(filename="test_dump.npy", n=2000, n_exp=200):
    from timeit import default_timer as timer
    from tqdm import tqdm

    start = timer()
    M = markov_chain(2)
    ns = [100, 1000, 2000]
    n_exp = 200

    d = 7
    N = n_exp // d

    pbar = tqdm(total=n_exp*sum(ns))

    def update(*a):
        pbar.update(N)

    pool = Pool()
    sims = []

    for n in ns:
        r = [pool.apply_async(run_simu_mr, (M, n, N), callback=update) for _ in range(d)]
        exps = sum([x.get() for x in r], [])
        sims.append(get_summary(exps))

    pbar.close()

    end = timer()
    print("Time taken:", end - start)
    np.save("tails_datas/" + filename, resu)

if __name__ == "__main__":
    import sys

    if sys.argv[1] == '-s':
        main(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))