from multiprocessing import Process, Queue, Pool
import numpy as np
import os
from math import log
from tails import run_experiment, get_summary
from markov import markov_chain, markov_iter_rng
import gc


def rng_gen(N, n):

    formula = int(N * 3 * n * log(n, 2))
    M = min(formula, 20_000_000)

    if formula > 20_000_000:
        print("Chose constant")
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

        tail_symbols = ""
        parse = set("")
        lnc = 0

        for k in range(1, n + 1):
            seq = ""
            m_iter = markov_iter_rng(M, rng)

            while seq in parse:
                seq += next(m_iter)

            parse.add(seq)
            tail_symbols += next(m_iter)

            lnc += len(seq)
            # Just keeping the total length lnc

        # Just keeping tnc
        exp["tnc"] = tail_symbols.count("0")
        exp["lnc"] = lnc

        exps.append(exp)

    exps[0]["M"] = M
    exps[0]["n"] = n
    exps[0]["n_exp"] = len(exps)

    return exps


def parallel_simu(M, ns, n_exp, d=4):
    from progress.bar import Bar

    N = n_exp // d

    pool = Pool()
    sims = []

    print("M is well defined : {}".format(M))
    bar = Bar("Number of DST built", max=len(ns) * (d * N))

    for n in ns:

        print("\n=============================================\n")
        print("Computing n = {} (max: {}), n_exp = {}".format(n, ns[-1], n_exp))

        def update(*a):
            bar.next(N)

        r = [
            pool.apply_async(run_simu_mr, (M, n, N), callback=update) for _ in range(d)
        ]

        exps = sum([x.get() for x in r], [])
        sims.append(get_summary(exps))

        print(
            "\nAverage total path length: {}".format(
                sims[-1][1]["variables"]["mean_lnc"]
            )
        )
        gc.collect()

    bar.finish()

    return sims


if __name__ == "__main__":

    from timeit import default_timer as timer
    from progress.bar import Bar
    import sys

    start = timer()
    M = markov_chain(2)

    if len(sys.argv) > 1:
        if sys.argv[1] == "-range":
            ns = list(
                range(int(input("a = ")), int(input("b = ")), int(input("step = ")))
            )
            n_exp = int(input("n_exp = "))

    else:
        ns = [100, 1000, 2000]
        n_exp = 300

    resu = parallel_simu(M, ns, n_exp)

    end = timer()
    print("Time taken:", end - start)
    # np.save("tails_datas/" + filename, resu)
    filename = "dummy"
    np.save("tails_datas/" + filename, resu)
    print("Saved in <tails_datas/{}>".format(filename))
