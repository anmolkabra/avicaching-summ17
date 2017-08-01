#!/usr/bin/env python

# =============================================================================
# lp.py
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Creates matrices and runs the LP for the Pricing Problem.
# -----------------------------------------------------------------------------
# Required Dependencies/Software:
#   - Python 2.x (obviously, Anaconda environment used originally)
#   - NumPy
#   - SciPy
# =============================================================================

from __future__ import print_function
import numpy as np
import scipy.optimize as sp_opt

# scipy standard LP format: min c^Tx s.t. Ax <= b

# make a linear program of this:
# argmin(sum(|| (r_i)_n - (r_i)_o||))
# s.t.
#     (r_i)_n >= 0
#     sum((r_i)_n) <= 1000

# l1 norm minimization converted to scipy supported lp:
# argmin(1^Tu)
# [  I_N   -I_N    ]              [  (r_i)_o  ]
# [ -I_N   -I_N    ] [(r_i)_n] <= [ -(r_i)_o  ]
# [ 1^T_N   0^T_N  ] [   u   ]    [   1000    ]
# r_i_n, u >= 0

# the objective function's weights c^T such that z = c^T . x are:
# [  0_N     1_N   ]

# N = 116

def build_A(N):
    """
    Build A based on the defined problem.

    Args:
        N -- (int) as defined above

    Returns:
        NumPy ndarray - A
    """
    A = np.hstack( (np.eye(N), np.negative(np.eye(N))) )
    A = np.vstack( (A, np.negative(np.hstack( (np.eye(N), np.eye(N)) ))) )
    A = np.vstack( (A, np.hstack( (np.ones(N), np.zeros(N)) )) )
    return A

def build_b(N, r_i_o, R):
    """
    Build b based on the defined problem.

    Args:
        N -- (int) as defined above
        r_i_o -- (NumPy ndarray) Old rewards vector
        R -- (float) total rewards

    Returns:
        NumPy ndarray - b
    """
    b = np.hstack( (r_i_o, np.negative(r_i_o), R) )
    return b

def build_c(N):
    """
    Build c based on the defined problem.

    Args:
        N -- (int) as defined above

    Returns:
        NumPy ndarray - c
    """
    return np.hstack( (np.zeros(N), np.ones(N)) )

def run_lp(A, c, N, r_i_o, R):
    """
    Returns the result of the LP problem.

    Args:
        A -- (NumPy ndarray) constructed using build_A()
        c -- (NumPy ndarray) constructed using build_c()
        N -- (int) as defined above
        r_i_o -- (NumPy ndarray) Old rewards vector
        R -- (float) total rewards

    Returns:
        scipy.optimize.OptimizeResult -- result of the scipy.optimize.linprog()
    """
    b = build_b(N, r_i_o, R)
    return sp_opt.linprog(c, A_ub=A, b_ub=b)   # default non negative bounds