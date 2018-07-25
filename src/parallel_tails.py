from multiprocessing import Process
import os
from stupid_tails import run_experiment

def worker():
    """worker function"""
    return run_experiment(10000)

if __name__ == '__main__':
    jobs = []
    for i in range(300):
        p = Process(target=worker)
        jobs.append(p)
        p.start()

    print("Done")
    for i in range(300):
        p.join()
    print("Done")