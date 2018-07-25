%%timeit

from multiprocessing import Process, Queue
import os
from stupid_tails import run_experiment
from markov import markov_chain, markov_iter

def run_experiment_multi(n, M, q):

    exp = dict()

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

    q.put(exp)

M = markov_chain(2)
n = 100
n_exp = 50
jobs = []
q = Queue()

def run_simu(n, M, N, q):
    for _ in range(N):
        run_experiment_multi(n, M, q)

N = n_exp // 4
for i in range(4):
    p = Process(target=run_simu, args=(n, M, N, q))
    jobs.append(p)
    p.start()

for j in jobs:
    j.join()
