#!/usr/bin/env python
from __future__ import print_function
import numpy as np, scipy.optimize as sp_opt

# make a test linear program of this:
# argmin(sum(|| (r_i)_n - (r_i)_o||))
# s.t.
#     (r_i)_n >= 0
#     sum((r_i)_n) <= 1000

# l1 norm minimization converted to scipy supported lp:
# argmin(1^T . u)
# [  I_N   -I_N    ]              [  (r_i)_o  ]
# [ -I_N   -I_N    ] [(r_i)_n] <= [ -(r_i)_o  ]
# [ 1^T_N   0^T_N  ] [   u   ]    [   1000    ]
# r_i_n, u >= 0

# the objective function's weights c^T such that z = c^T . x are:
# [  0_N     1_N   ]

# N = 116

def build_A(N):
    """
    build A based on the defined problem
    """
    A = np.hstack( (np.eye(N), np.negative(np.eye(N))) )
    A = np.vstack( (A, np.negative(np.hstack( (np.eye(N), np.eye(N)) ))) )
    A = np.vstack( (A, np.hstack( (np.ones(N), np.zeros(N)) )) )
    return A

def build_b(N, r_i_o, R):
    """
    build b based on the defined problem
    """
    b = np.hstack( (r_i_o, np.negative(r_i_o), R) )
    return b

def build_c(N):
    return np.hstack( (np.zeros(N), np.ones(N)) )

def run_lp(N, r_i_o, R):
    """
    returns the result of the lp problem
    """
    A = build_A(N)
    b = build_b(N, r_i_o, R)
    c = build_c(N)
    return sp_opt.linprog(c, A_ub=A, b_ub=b)   # default non negative bounds

# rio = np.random.randn(5)
# print(sum(rio))
# res = run_lp(5, rio, 4)
# print(res)
# print(sum(res.x[:5]))