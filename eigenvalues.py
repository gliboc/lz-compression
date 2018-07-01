r"""Computing the eigenvalues expression of the variance 
from \cite{avg}"""

from markov import markov_chain
from szpan import entropy
import numpy.linalg
from math import sqrt, log
import numpy as np


def eigenvalue(M):

    p00 = M[0, 0]
    p01 = M[0, 1]
    p10 = M[1, 0]
    p11 = M[1, 1]

    e_vect = numpy.linalg.eig(M)

    pi_0 = p10 / (p01 + p10)
    pi_1 = p01 / (p01 + p10)

    h = entropy(M)

    delta = p00 * p11 - p01 * p10
    der_delta = - log( p00 * p11 ) * (p00 * p11) + log( p01 * p10 ) * (p01 * p10)
    
    g_0 = (- log(p11) * p11 + log(p01) * p01) * delta \
            - (p11 - p01) * der_delta

    g_1 = (- log(p00) * p00 + log(p10) * p10) * delta \
            + (p00 - p10) * der_delta

    der_pi_0 = h * (p11 - p01) / delta + g_0 / (delta ** 2)
    der_pi_1 = h * (p00 - p10) / delta + g_1 / (delta ** 2)

    return der_pi_0, der_pi_1


def lambda_2(M):

    dp0, dp1 = eigenvalue(M)

    p00 = M[0, 0]
    p01 = M[0, 1]
    p10 = M[1, 0]
    p11 = M[1, 1]

    pi_0 = p10 / (p01 + p10)
    pi_1 = p01 / (p01 + p10)

    h = entropy(M)

    t0 = pi_0 * log(p00) ** 2 * p00 + pi_1 * log(p10) ** 2 * p10
    t0 += pi_0 * log(p01) ** 2 * p01 + pi_1 * log(p11) ** 2 * p11

    t1 = dp0 * p00 * log(p00) + dp1 * p10 * log(p10)
    t1 += dp0 * p01 * log(p01) + dp1 * p11 * log(p11)
    t1 *= -2

    t2 = - 2 * h * (dp0 + dp1)

    return t0 + t1 + t2


if __name__ == "__main__":
    import pandas as pd 

    comps = []
    las = []
    for _ in range(10):
        M = markov_chain(2)
        comps.append(eigenvalue(M))
        las.append(lambda_2(M))


    dpi0s = [c[0] for c in comps]
    dpi1s = [c[1] for c in comps]

    d = {"dpi0" : dpi0s,
         "dpi1" : dpi1s,
         "ddlambda": las}

    df = pd.DataFrame(data=d)

    print(df) 

    with open("d_pi_table.dat", "w") as fo:
        x = df.to_latex()
        for l in x:
            fo.write(l)

