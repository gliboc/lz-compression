from math import log

def diff(p):
    q = 1-p

    def h(p):
        return -p * log(p) - (1-p) * log (1-p)

    return (q/p) * h(p)**2 + 2*q*h(p)*log(q) - 2*q*h(p)*log(p)
