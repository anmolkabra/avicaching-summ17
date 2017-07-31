#!/usr/bin/env python

# =============================================================================
# avicaching_data.py
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Reads, writes, mutates and operates on avicaching data files. This module 
#   is imported in all models for reading and doing things.
# -----------------------------------------------------------------------------
# What do terms mean:
#   X - visit densities before placing rewards (for Identification Problem)
#   Y - visit densities after placing rewards (for Identification Problem)
#   R - rewards matrix
#   J - no. of locations
#   T - no. of time units
# -----------------------------------------------------------------------------
# Required Dependencies/Software:
#   - Python 2.x (obviously, Anaconda environment used originally)
#   - NumPy
# -----------------------------------------------------------------------------
# Required Local Files/Data/Modules:
#   - ./data/*
# =============================================================================

from __future__ import print_function
import numpy as np

def read_XYR_file(file_name, locs, T):
    """
    Read the datafile containing X, Y, R information.

    Args:
        file_name -- (str) name of the file
        locs -- (int) J 
        T -- (int) T

    Returns:
        3-tuple -- (tuple of NumPy ndarrays) X, Y, R
    """
    X, Y, R = [], [], []
    with open(file_name, "r") as xyrfile:
        for idx, line in zip(xrange(T * 3), xyrfile):
            # the line contains locs+ floats separated by spaces
            line_vec = np.array(map(float, line.split()[:locs]))   # only J cols
            # if reading the first three lines, init the arrays
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
                # reading 4+ lines
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

def read_F_file(file_name, locs):
    """
    Reads the csv file containing f information.

    Args:
        file_name -- (str) name of the file
        locs -- (int) J

    Returns:
        NumPy ndarray -- f
    """
    f = []
    with open(file_name, "r") as fufile:
        next(fufile)        # skip header row of fufile
        for idx, line in zip(xrange(locs), fufile):
            # ignore last 3 cols which contain lat long information
            line_vec = np.array(map(float, line.split(",")[:-3]))
            if idx == 0:
                # F init
                f = line_vec
            else:
                # append F info
                f = np.vstack([f, line_vec])
    return f

def read_dist_file(file_name, locs):
    """
    Reads the DIST file containing the distances between all locations.

    Args:
        file_name -- (str) name of the file
        locs -- (int) J

    Returns:
        NumPy ndarray -- DIST
    """
    DIST = []
    with open(file_name, "r") as distfile:
        for idx, line in zip(xrange(locs), distfile):
            # only read J rows and cols
            line_vec = np.array(map(float, line.split()))[:locs]
            if idx == 0:
                # DIST init
                DIST = line_vec
            else:
                # append DIST info
                DIST = np.vstack([DIST, line_vec])
    return DIST

def combine_DIST_F(f, DIST, locs, numFeatures):
    """
    Combines f and DIST as data preprocessing.

    Args:
        F -- (NumPy ndarray) f
        DIST -- (NumPy ndarray) DIST
        locs -- (int) J
        numFeatures -- (int) `len(f[i]) + 1` (accounting for the distance element)

    Returns:
        NumPy ndarray -- represents the input dataset without the rewards.
    """
    NN_in = np.empty([locs, locs, numFeatures], dtype=float)
    for v in xrange(locs):
        for u in xrange(locs):
            NN_in[v][u][0] = DIST[v][u]
            NN_in[v][u][1:] = f[u]
    return NN_in

def save_rand_XYR(file_name, X, Y, R, J=116, T=173):
    """
    Writes X, Y, R information to a file such that it is readable by read_XYR_file().
    The dimensions of X, Y, R must be J x T each.

    Args:
        file_name -- (str) name of the file
        X -- (NumPy ndarray) X
        Y -- (NumPy ndarray) Y
        R -- (NumPy ndarray) R
        J -- (int) no. of locations (default=116)
        T -- (int) no. of time units (default=173)
    """
    # to intersperse XYR
    XYR = np.empty([T * 3, J])
    
    for t in xrange(T):
        if t == 0:
            XYR = X[t]
        else:
            XYR = np.vstack([XYR, X[t]])
        XYR = np.vstack([XYR, Y[t]])
        XYR = np.vstack([XYR, R[t]])
    
    np.savetxt(file_name, XYR, fmt="%.8f", delimiter=" ")

def split_along_dim(M, num, dim):
    """
    Shuffles and splits a matrix into 2 along a dimension at index, returning
    two matrices.

    Args:
        M -- (NumPy ndarray) Matrix to be split
        num -- (int) index to split at
        dim -- (int) axis/dimension for splitting

    Returns:
        2 NumPy ndarrays -- Matrices after splitting
    """
    return np.split(M, [num], axis=dim)

def read_weights_file(file_name, locs_in_file, locs, numFeatures):
    """
    Reads the weights file and splits the saved data into 2 weights tensors, 
    as our models require.

    Args:
        file_name -- (str) name of the file
        locs_in_file -- (int) no. of locations used in the weights file
        locs -- (int) no. of locations required by the run specs 
            (<= locs_in_file). The function reads the file and ignores data 
            corresponding to locations > locs
        numFeatures -- (int) no. of features in the dataset (the run spec 
            should not request something different than what is in the file). 
            This also determines the size of the weights tensors

    Returns:
        2-tuple of NumPy ndarrays -- w1 and w2
    """
    data = np.loadtxt(file_name)
    # data is an ndarray with the 1st part representing the 3d w1 (represented as 
    # 2d slices) and the 2nd part representing w2 - the last locs_in_file rows
    w1, w2 = split_along_dim(data, len(data) - locs_in_file, dim=0)
    # take out only locs slices. Since w1 is represented in 2d, this means 
    # taking out locs * numFeatures slices
    w1, w2 = w1[:locs * numFeatures], w2[:locs]
    w1 = w1.reshape((locs, numFeatures, numFeatures))
    return (w1, w2)

def read_lat_long_from_Ffile(file_name, locs, lat_col=33, long_col=34):
    """
    Reads the latitude and longitude from the file containing f information.

    Args:
        file_name -- (str) name of the file
        locs -- (int) no. of locations. Also represents the length of lat and 
            long vectors
        lat_col -- (int) col no. in the f file
        long_col -- (int) col no. in the f file

    Returns:
        NumPy ndarray -- 2d matrix where the first col are latitudes and the 
        second col are longitudes
    """
    lat_long = []
    with open(file_name, "r") as fufile:
        next(fufile)        # skip header row of fufile
        for idx, line in zip(xrange(locs), fufile):
            # extract latitude and longitude. Since they are stored adjacent, 
            # just ignore other cols
            line_vec = np.array(map(float, line.split(",")[lat_col:long_col + 1]))
            if idx == 0:
                # lat_long init
                lat_long = line_vec
            else:
                # append lat_long info
                lat_long = np.vstack([lat_long, line_vec])
    return lat_long

def normalize(x, along_dim=None, using_max=True, offset_division=0.000001):
    """
    Normalizes a tensor by dividing each element by the maximum or by the sum, 
    which are calculated along a dimension. 

    Args:
        x -- (NumPy ndarray) matrix/tensor to be normalized
        along_dim -- (int or None) If along_dim is an int, the max is 
            calculated along that dimension; if it's None, 
            whole x's max/sum is calculated (default=None)
        using_max -- (bool) Normalize using max if True and sum if False 
            (default=True)
        offset_division -- (float) safety mechanism to avoid division by zero

    Returns:
        NumPy ndarray -- Normalized matrix/tensor
    """
    if using_max:
        return x / (np.amax(x, axis=along_dim) + offset_division)
    else:
        return x / (np.sum(x, axis=along_dim, keepdims=True) + offset_division)

def make_rand_F_file(file_name, J):
    """
    [Extremely bad code. A very bad example of coding style]
    Creates and write random f file.

    Args:
        file_name -- (str) name of the file
        J -- (int) J
    """
    # num visits  -- type random int
    num_visits = np.floor(np.random.rand(J) * 1000)
    # num species -- type random int
    num_species = np.floor(np.random.rand(J) * 500)
    # NLCD2011_FS_C11_375_PLAND -- type random float
    NLCD2011_FS_C11_375_PLAND = np.random.rand(J) * 100
    # NLCD2011_FS_C12_375_PLAND -- zeros
    NLCD2011_FS_C12_375_PLAND = np.zeros(J)
    # NLCD2011_FS_C21_375_PLAND -- type random float
    NLCD2011_FS_C21_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C22_375_PLAND -- type random float
    NLCD2011_FS_C22_375_PLAND = np.random.rand(J) * 50
    # NLCD2011_FS_C23_375_PLAND -- type random float
    NLCD2011_FS_C23_375_PLAND = np.random.rand(J) * 50
    # NLCD2011_FS_C24_375_PLAND -- type random float
    NLCD2011_FS_C24_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C31_375_PLAND -- type random float
    NLCD2011_FS_C31_375_PLAND = np.random.rand(J) * 2
    # NLCD2011_FS_C41_375_PLAND -- type random float
    NLCD2011_FS_C41_375_PLAND = np.random.rand(J) * 100
    # NLCD2011_FS_C42_375_PLAND -- type random float
    NLCD2011_FS_C42_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C43_375_PLAND -- type random float
    NLCD2011_FS_C43_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C52_375_PLAND -- type random float
    NLCD2011_FS_C52_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C71_375_PLAND -- type random float
    NLCD2011_FS_C71_375_PLAND = np.random.rand(J) * 2
    # NLCD2011_FS_C81_375_PLAND -- type random float
    NLCD2011_FS_C81_375_PLAND = np.random.rand(J) * 100
    # NLCD2011_FS_C82_375_PLAND -- type random float
    NLCD2011_FS_C82_375_PLAND = np.random.rand(J) * 80
    # NLCD2011_FS_C90_375_PLAND -- type random float
    NLCD2011_FS_C90_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C95_375_PLAND -- type random float
    NLCD2011_FS_C95_375_PLAND = np.random.rand(J) * 2
    # HOUSING_DENSITY -- type random float
    HOUSING_DENSITY = np.random.rand(J) * 500
    # HOUSING_PERCENT_VACANT -- type random float
    HOUSING_PERCENT_VACANT = np.random.rand(J) * 0.1
    # ELEV_GT -- type random int
    ELEV_GT = np.floor(np.random.rand(J) * 500)
    # DIST_FROM_FLOWING_FRESH -- type random int
    DIST_FROM_FLOWING_FRESH = np.floor(np.random.rand(J) * 5)
    # DIST_IN_FLOWING_FRESH -- type random int
    DIST_IN_FLOWING_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_STANDING_FRESH -- type random int
    DIST_FROM_STANDING_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_STANDING_FRESH -- type random int
    DIST_IN_STANDING_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_WET_VEG_FRESH -- type random int
    DIST_FROM_WET_VEG_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_WET_VEG_FRESH -- type random int
    DIST_IN_WET_VEG_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_FLOWING_BRACKISH -- type random int
    DIST_FROM_FLOWING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_FLOWING_BRACKISH -- type random int
    DIST_IN_FLOWING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_STANDING_BRACKISH -- type random int
    DIST_FROM_STANDING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_STANDING_BRACKISH -- type random int
    DIST_IN_STANDING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_WET_VEG_BRACKISH -- type random int
    DIST_FROM_WET_VEG_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_WET_VEG_BRACKISH -- type random int
    DIST_IN_WET_VEG_BRACKISH = np.floor(np.random.rand(J) * 10)
    # LATITUDE -- type intersperse between 42 44
    LATITUDE = np.linspace(42, 44, num=J)
    # LONGITUDE --type intersperse between -75 -77
    LONGITUDE = np.linspace(-75, -77, num=J)
    # LOC_ID --random
    LOC_ID = np.random.rand(J)


    ###
    data = np.vstack([num_visits,
        num_species,
        NLCD2011_FS_C11_375_PLAND,
        NLCD2011_FS_C12_375_PLAND,
        NLCD2011_FS_C21_375_PLAND,
        NLCD2011_FS_C22_375_PLAND,
        NLCD2011_FS_C23_375_PLAND,
        NLCD2011_FS_C24_375_PLAND,
        NLCD2011_FS_C31_375_PLAND,
        NLCD2011_FS_C41_375_PLAND,
        NLCD2011_FS_C42_375_PLAND,
        NLCD2011_FS_C43_375_PLAND,
        NLCD2011_FS_C52_375_PLAND,
        NLCD2011_FS_C71_375_PLAND,
        NLCD2011_FS_C81_375_PLAND,
        NLCD2011_FS_C82_375_PLAND,
        NLCD2011_FS_C90_375_PLAND,
        NLCD2011_FS_C95_375_PLAND,
        HOUSING_DENSITY,
        HOUSING_PERCENT_VACANT,
        ELEV_GT,
        DIST_FROM_FLOWING_FRESH,
        DIST_IN_FLOWING_FRESH,
        DIST_FROM_STANDING_FRESH,
        DIST_IN_STANDING_FRESH,
        DIST_FROM_WET_VEG_FRESH,
        DIST_IN_WET_VEG_FRESH,
        DIST_FROM_FLOWING_BRACKISH,
        DIST_IN_FLOWING_BRACKISH,
        DIST_FROM_STANDING_BRACKISH,
        DIST_IN_STANDING_BRACKISH,
        DIST_FROM_WET_VEG_BRACKISH,
        DIST_IN_WET_VEG_BRACKISH,
        LATITUDE,
        LONGITUDE,
        LOC_ID])

    with open(file_name, "w") as f:
        f.write("num visits,num species,NLCD2011_FS_C11_375_PLAND,NLCD2011_FS_C12_375_PLAND,NLCD2011_FS_C21_375_PLAND,NLCD2011_FS_C22_375_PLAND,NLCD2011_FS_C23_375_PLAND,NLCD2011_FS_C24_375_PLAND,NLCD2011_FS_C31_375_PLAND,NLCD2011_FS_C41_375_PLAND,NLCD2011_FS_C42_375_PLAND,NLCD2011_FS_C43_375_PLAND,NLCD2011_FS_C52_375_PLAND,NLCD2011_FS_C71_375_PLAND,NLCD2011_FS_C81_375_PLAND,NLCD2011_FS_C82_375_PLAND,NLCD2011_FS_C90_375_PLAND,NLCD2011_FS_C95_375_PLAND,HOUSING_DENSITY,HOUSING_PERCENT_VACANT,ELEV_GT,DIST_FROM_FLOWING_FRESH,DIST_IN_FLOWING_FRESH,DIST_FROM_STANDING_FRESH,DIST_IN_STANDING_FRESH,DIST_FROM_WET_VEG_FRESH,DIST_IN_WET_VEG_FRESH,DIST_FROM_FLOWING_BRACKISH,DIST_IN_FLOWING_BRACKISH,DIST_FROM_STANDING_BRACKISH,DIST_IN_STANDING_BRACKISH,DIST_FROM_WET_VEG_BRACKISH,DIST_IN_WET_VEG_BRACKISH,LATITUDE,LONGITUDE,LOC_ID\n")
        np.savetxt(f, data.T, fmt="%.5f", delimiter=",")

def make_rand_DIST_file(file_name, J):
    """
    J x J random matrix max 100. diagonal elements 0
    """
    data = np.random.rand(J, J) * 100
    data[np.diag_indices(J)] = 0.0
    np.savetxt(file_name, data, fmt="%.6f", delimiter=" ")

def combine_lp_time_log(outfile, cpu_set, gpu_set, onlylp):
    """
    combines lp runtime logs for tex-tikz input
    """
    with open(cpu_set, "r") as c, open(gpu_set, "r") as g, open(onlylp) as o,\
        open(outfile, "w") as out:
        out.write("epoch,cpuset,gpuset,onlylp\n")
        e = 1
        for cline, gline, oline in zip(c, g, o):
            out.write("%d,%.6f,%.6f,%.6f\n" % \
                (e, float(cline.split(",")[1]), float(gline.split(",")[1]), float(oline.split(",")[1])))
            e += 1

def combine_lp_time_log_threads(outfile, thread1, thread3, thread5, thread7):
    """
    combines lp runtime logs for tex-tikz input
    """
    with open(thread1, "r") as t1, open(thread3, "r") as t3, open(thread5) as t5,\
        open(thread7, "r") as t7, open(outfile, "w") as out:
        out.write("epoch,t1,t3,t5,t7\n")
        e = 1
        for t1line, t3line, t5line, t7line in zip(t1,t3,t5,t7):
            out.write("%d,%.6f,%.6f,%.6f,%.6f\n" % \
                (e, float(t1line.split(",")[1]), float(t3line.split(",")[1]), \
                    float(t5line.split(",")[1]), float(t7line.split(",")[1])))
            e += 1

def extract_python_processes(outfile, infile):
    with open(infile, "r") as i, open(outfile, "w") as o:
        e = 1
        o.write("epoch,cpu,mem\n")
        for line in i:
            if "python" in line:
                # pid, mem, cpu, name -- order of stored info
                el = line.split(" ")
                # print(elements[1], elements[2], elements[3])
                o.write("%d,%.1f,%.1f\n" % \
                    (e, float(el[1]), float(el[2])))
                e += 1
