from __future__ import print_function
import numpy as np

def read_XYR_file(file_name, locs, T):
    X, Y, R = [], [], []
    with open(file_name, "r") as xyrfile:
        for idx, line in zip(xrange(T * 3), xyrfile):
            line_vec = np.array(map(float, line.split()[:locs]))   # only J cols
            if idx == 0:
                # X init
                X = line_vec
            elif idx == 1:
                # Y init
                Y = line_vec
            elif idx == 2:
                # R init
                R = line_vec
            else:
                rem = idx % 3
                if rem == 0:
                    # append X information
                    X = np.vstack([X, line_vec])
                elif rem == 1:
                    # append Y information
                    Y = np.vstack([Y, line_vec])
                else:
                    # append R information
                    R = np.vstack([R, line_vec])
    return (X, Y, R)

def read_F_file(file_name):
    F = []
    with open(file_name, "r") as fufile:
        next(fufile)        # skip header row of fufile
        for idx, line in enumerate(fufile):
            line_vec = np.array(map(float, line.split(",")[:-3]))  # ignore last 3 cols
            if idx == 0:
                # F init
                F = line_vec
            else:
                # append F info
                F = np.vstack([F, line_vec])
    return F

def read_dist_file(file_name, locs):
    DIST = []
    with open(file_name, "r") as distfile:
        for idx, line in zip(xrange(locs), distfile):
            line_vec = np.array(map(float, line.split()))[:locs]
            if idx == 0:
                # DIST init
                DIST = line_vec
            else:
                # append DIST info
                DIST = np.vstack([DIST, line_vec])
    return DIST

def combine_DIST_F(F, DIST, locs, numFeatures):
    NN_in = np.empty([locs, locs, numFeatures], dtype=float)
    for v in xrange(len(DIST)):
        for u in xrange(len(DIST[0])):
            NN_in[v][u][0] = DIST[v][u]
            NN_in[v][u][1:] = F[u]    # last one reserved for rewards
    return NN_in

def make_rand_XYR(file_name, J=116, T=173, X_max=10.0, Y_max=10.0, R_max=10.0):
    X = np.floor(np.random.rand(T, J) * X_max)
    Y = np.floor(np.random.rand(T, J) * Y_max)
    R = np.floor(np.random.rand(T, J) * R_max)

    # intersperse XYR
    XYR = np.empty([T * 3, J])
    for t in xrange(T):
        if t == 0:
            XYR = X[t]
            XYR = np.vstack([XYR, Y[t]])
            XYR = np.vstack([XYR, R[t]])
        else:
            XYR = np.vstack([XYR, X[t]])
            XYR = np.vstack([XYR, Y[t]])
            XYR = np.vstack([XYR, R[t]])
    
    np.savetxt(file_name, XYR, fmt="%.1f", delimiter=" ")