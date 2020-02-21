import numpy as np

DEPTH = 2
DEF_A = 0.80


def h_a(x, a=DEF_A):
    assert 0 <= a <= 1
    eps = 1 if 2*x > 1 else -1
    return 1/2 + (1/2)*eps*np.power(np.abs(2*x-1), (1-a)/a)


def h_a_sep(x, sep, a=DEF_A):
    if x >= sep:
        v = ((x-sep)/(1-sep))*0.5 + 0.5
        return h_a(v, a=a)
    v = (x/sep)*0.5
    return h_a(v, a=a)


def get_etas(x, a=DEF_A):
    probs = []
    def rec(sep, depth, a=a):
        inter = h_a_sep(x, sep, a=a)
        if depth == 0:
            return [1-inter, inter]
        split_size = 1/2**(DEPTH+2-depth)
        ln = rec(sep - split_size, depth-1, a)
        rn = rec(sep + split_size, depth-1, a)
        return [(1-inter)*a for a in ln] + [inter*a for a in rn]
    return rec(1/2, DEPTH)
