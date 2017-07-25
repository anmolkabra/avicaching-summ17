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

def read_F_file(file_name, locs):
    F = []
    with open(file_name, "r") as fufile:
        next(fufile)        # skip header row of fufile
        for idx, line in zip(xrange(locs), fufile):
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

def save_rand_XYR(file_name, X, Y, R, J=116, T=173):
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
    
    np.savetxt(file_name, XYR, fmt="%.8f", delimiter=" ")

def split_along_dim(M, num, dim):
    """
    Shuffles and splits the matrix M into 2 along dimension 0 at index num, returning
    two matrices
    """
    return np.split(M, [num], axis=dim)

def read_weights_file(file_name, locs_in_file, locs, numFeatures):
    data = np.loadtxt(file_name)
    w1, w2 = split_along_dim(data, len(data) - locs_in_file, dim=0)  # w2 is the last locs_in_file rows
    w1, w2 = w1[:locs * numFeatures], w2[:locs] # take out only locs slices
    w1 = w1.reshape((locs, numFeatures, numFeatures))
    return (w1, w2)

def read_lat_long_from_Ffile(file_name, locs, lat_col=33, long_col=34):
    lat_long = []
    with open(file_name, "r") as fufile:
        next(fufile)        # skip header row of fufile
        for idx, line in zip(xrange(locs), fufile):
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
    Normalizes x by dividing each element by the maximum if using_max is True and by the sum
    if using_max is False. Finding the maximum/sum is specified by along_dim. If along_dim is an int, the max is 
    calculated along that dimension, if it's None, whole x's max/sum is calculated
    """
    if using_max:
        return x / (np.amax(x, axis=along_dim) + offset_division)
    else:
        return x / (np.sum(x, axis=along_dim, keepdims=True) + offset_division)

def make_rand_F_file(file_name, J):
    """
    I admit this is very bad code. I didn't care about optimizing this data
    constructor
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
