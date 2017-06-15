#!/usr/bin/env python
from __future__ import print_function
import os, numpy as np

def read_first_k_lines(fname, k):
    w = []
    with open(fname, "r") as f:
        for i in xrange(k):
            line_vec = np.array(map(float, f.readline().split()))
            if i == 0:
                # w init
                w = line_vec
            else:
                # append w info
                w = np.vstack([w, line_vec])
    return w

orig_weights = np.loadtxt("randXYR_weights.txt")
weights_files = os.listdir("./recovering_weights/")  # contains a list of file names
for file in weights_files:
    weights_before = read_first_k_lines("./recovering_weights/" + file, 116)
    weights_after = np.loadtxt("./recovering_weights/" + file, skiprows=10116)
    # print(weights_before.shape, weights_before)
    # print(weights_after.shape, weights_after)
    
    dist_before = np.linalg.norm(orig_weights - weights_before)
    dist_after = np.linalg.norm(orig_weights - weights_after)
    print("norm of diff -- before: %.8f, after: %.8f" % (dist_before, dist_after))
    # with open("./recovering_weights/" + file, "r") as f:
    #     data = read_weights_file()
    #     data = f.readlines()