r"""Computing the eigenvalues expression of the variance 
from \cite{avg}"""

from markov import markov_chain
from szpan import entropy
import numpy.linalg
from math import sqrt, log
import numpy as np

# based on wrong computations
# def dpi(M):

#     p00 = M[0, 0]
#     p01 = M[0, 1]
#     p10 = M[1, 0]
#     p11 = M[1, 1]

    # e_vect = numpy.linalg.eig(M)

    # pi_0 = p10 / (p01 + p10)
    # pi_1 = p01 / (p01 + p10)

    # h = entropy(M) # ok

    # d1 = p00 * p11 - p01 * p10 # ok
    # dd1 = - log( p00 * p11 ) * p00 * p11 + log( p01 * p10) * p01 * p10 # ok

    # g0 = p11 - p01 # ok
    # dg0 = - log( p11 ) * p11 + log( p01 ) * p01 # ok

    # g1 = p00 - p10 # ok
    # dg1 = - log( p00 ) * p00 + log( p10 ) * p10 # ok

    # dpi0 = h * g0 / d1 + (dg0 * d1 - g0 * dd1) / (d1 ** 2) # ok
    # dpi1 = h * g1 / d1 + (dg1 * d1 - g1 * dd1) / (d1 ** 2) # ok

    # return dpi0, dpi1


def dpi2(M):
    p00 = M[0, 0]
    p01 = M[0, 1]
    p10 = M[1, 0]
    p11 = M[1, 1]

    pi_0 = p10 / (p01 + p10)
    pi_1 = p01 / (p01 + p10)

    return (-log(pi_0) * pi_0, -log(pi_1) * pi_1)

def lambda_2(M):

    dp0, dp1 = dpi2(M)

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

from math import sqrt



def compute_lambda(M):
    p00 = M[0, 0]
    p01 = M[0, 1]
    p11 = M[1, 1]
    p10 = M[1, 0]

    q0 = p00 * p11
    q1 = p01 * p10

    alpha = p00 ** 2 + p11 ** 2 - 2 * q0 + 4 * q1

    der_alpha = - 2 * log(p00) * (p00 ** 2) - 2 * log(p11) * (p11 ** 2) \
                + 2 * q0 * log(q0) \
                - 4 * q1 * log(q1)

    der2_alpha = 4 * (log(p00) ** 2) * (p00 ** 2) \
                 + 4 * (log(p11) ** 2) * (p11 ** 2) \
                 - 2 * (log(q0) ** 2) * q0 \
                 + 4 * (log(q1) ** 2) * q1

    beta = 0
    der_beta = 0
    der2_beta = 0

    gamma = der_alpha * alpha # beta terms are zero
    der_gamma = der2_alpha * alpha + (der_alpha ** 2) # beta terms are zero

    kappa = sqrt(alpha**2 + beta**2) # basically alpha
    der_kappa = (alpha * der_alpha) / (sqrt(alpha**2))

    f = 0.5 * (sqrt(alpha ** 2 + beta ** 2) + alpha)
    der_f = 0.5 * ((der_alpha * alpha) / (sqrt(alpha**2)) + der_alpha)
    der2_f = 0.5 * ( (der_gamma * kappa - gamma * der_kappa) / (kappa ** 2) + der2_alpha )

    x = sqrt(alpha)
    der_x = der_f / (2 * x)
    der2_x = (der2_f * x - der_f * der_x) / (2 * (x**2))

    lamb = 0.5 * (p00 + p11 + x)
    der_lamb = 0.5 * (- log(p00) * p00 - log(p11) * p11 + der_x)
    der2_lamb = 0.5 * ( (log(p00)**2) * p00 + (log(p11)**2) * p11 + der2_x )

    # print(alpha)
    # print(der_alpha)

    # print(f)
    # print(der_f)

    # print(x)
    # print(der_x)
    # print(der2_x)

    o = M[1, 0] + M[0, 1]
    p = [M[1, 0] / o, M[0, 1] / o]

    h = 0
    n = len(M)

    for i in range(n):
        for j in range(n):

            h -= p[i] * M[i, j] * log(M[i, j])

    # var_coeff = (der2_lamb - der_lamb ** 2) / (der_lamb**3)
    var_coeff = (der2_lamb - der_lamb ** 2) # the h^3 leaves
    # print("lambda", lamb)
    # print("der_lamb", der_lamb)
    # print("entropy", h)
    # print("der2_lamb", der2_lamb)
    # print("variance_constant_coeff", (var_coeff))
    print(der_lamb, h)
    assert(abs(der_lamb - h) < 1e-6)

    return var_coeff


def eigenvalue_std(M, n):
    v_coeff = compute_lambda(M)
    h = entropy(M)

    return sqrt(n * v_coeff) / log(n, 2) # - sqrt(n) * (40 / (1000 * (sqrt(5) - 1) ) )



if __name__ == "__main__":
    import pandas as pd 

    comps = []
    las = []
    v_coeffs = []
    vs = []
    stds = []

    for _ in range(10):
        M = markov_chain(2)
        comps.append(dpi2(M))
        h = entropy(M) 
        la = lambda_2(M)
        las.append(la)
        vs.append((la-h**2)/(h**3))
        v_coeffs.append(compute_lambda(M))
        stds.append(eigenvalue_std(M, 500))


    dpi0s = [c[0] for c in comps]
    dpi1s = [c[1] for c in comps]

    d = {"dpi0": dpi0s,
         "dpi1": dpi1s,
         "ddlambda": las,
         "vars": vs,
         "var_coeffs": v_coeffs,
         "stds": stds}

    df = pd.DataFrame(data=d)

    print(df) 

    with open("d_pi_table.dat", "w") as fo:
        x = df.to_latex()
        for l in x:
            fo.write(l)

