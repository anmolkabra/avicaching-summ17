#!/usr/bin/env python
from __future__ import print_function
import argparse, time, math, os, sys, numpy as np, matplotlib
try:
    os.environ["DISPLAY"]
except KeyError as e:
    # working without X/GUI environment
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import avicaching_data as ad
# import torch packages
import torch, torch.nn as nn, torch.nn.functional as torchfun, torch.optim as optim
from torch.autograd import Variable

# =============================================================================
# training specs
# =============================================================================
parser = argparse.ArgumentParser(description="NN Avicaching model for finding weights")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
    help="inputs learning rate of the network (default=0.01)")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=10, metavar="E",
    help="inputs the number of epochs to train for")
parser.add_argument("--locations", type=int, default=116, metavar="J",
    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int, default=173, metavar="T",
    help="inputs total time of data collection; number of weeks (default=173)")
parser.add_argument("--eta", type=float, default=10.0, metavar="F",
    help="inputs parameter eta in the model (default=10.0)")
parser.add_argument("--lambda-L1", type=float, default=10.0, metavar="LAM",
    help="inputs the L1 regularizing coefficient")
parser.add_argument("--rand", action="store_true", default=False,
    help="uses random xyr data")
parser.add_argument("--log-interval", type=int, default=1, metavar="I",
    help="prints training information at I epoch intervals (default=1)")
parser.add_argument("--hide-loss-plot", action="store_true", default=False,
    help="hides the loss plot, which is only saved")
parser.add_argument("--hide-map-plot", action="store_true", default=False,
    help="hides the map plot, which is only saved")
parser.add_argument("--expand-R", action="store_true", default=False,
    help="expands the reward vectors into matrices with distributed rewards")
parser.add_argument("--train-percent", type=float, default=0.8, metavar="T",
    help="breaks the data into T percent training and rest testing (default=0.8)")
parser.add_argument('--seed', type=int, default=1, metavar='S',
    help='random seed (default=1)')

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.should_test = (args.train_percent != 1.0)

torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =============================================================================
# constants and parameters
# =============================================================================
torchten = torch.FloatTensor
J, T, numFeatures = args.locations, args.time, 0
trainX, trainY, trainR, testX, testY, testR, F_DIST = [], [], [], [], [], [], []
u_train, u_test = np.array([]), np.array([])

num_train = int(math.floor(args.train_percent * T))
num_test = T - num_train

randXYR_file = "./data/random/randXYR" + str(J) + ".txt"
randXYR_weights_file = "./data/random/randXYR" + str(J) + "_weights.txt"
randF_file = "./data/random/randF" + str(J) + ".csv"
randDIST_file = "./data/random/randDIST" + str(J) + ".txt"

# =============================================================================
# data input functions
# =============================================================================
def read_set_data():
    """
    Reads and sets up the datasets
    """
    global trainX, trainY, trainR, testX, testY, testR, F_DIST, numFeatures
    global u_train, u_test
    # read f and dist datasets from file, operate on them
    if args.rand:
        F = ad.read_F_file(randF_file, J)
        DIST = ad.read_dist_file(randDIST_file, J)
    else:
        F = ad.read_F_file("./data/loc_feature_with_avicaching_combined.csv", J)
        DIST = ad.read_dist_file("./data/site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)
    F, DIST = ad.normalize(F, along_dim=0, using_max=True), ad.normalize(DIST, using_max=True)  # normalize using max
    
    # process data for the NN
    numFeatures = len(F[0]) + 1     # distance included
    F_DIST = torchten(ad.combine_DIST_F(F, DIST, J, numFeatures))
    numFeatures += 1                # for rewards later

    X, Y, R = [], [], []
    # operate on XYR data
    if args.rand:
        if not os.path.isfile(randXYR_file):
            print('a')
            # file doesn't exists, make random data, write to file
            X, Y, R = make_rand_data()
            ad.save_rand_XYR(randXYR_file, X, Y, R, J, T)
            sys.exit()
        #
        # print("Verifying randXYR...")
        # X, Y, R = ad.read_XYR_file("./randXYR.txt", J, T)
        # w = ad.read_weights_file("./randXYR_weights.txt", J)
        # X, Y, R, w = Variable(torchten(X)), Variable(torchten(Y)), torchten(R), torchten(w)
        # w = Variable(torch.unsqueeze(w, dim=2))   # make w a 3d tensor
        
        # test_given_data(X, Y, R, w, J, T)
        #
        X, Y, R = ad.read_XYR_file(randXYR_file, J, T)
    else:
        X, Y, R = ad.read_XYR_file("./data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)
    
    u = np.sum(Y, axis=1)   # u weights for calculating losses

    # normalize X, Y using sum along rows
    X, Y = ad.normalize(X, along_dim=1, using_max=False), ad.normalize(Y, along_dim=1, using_max=False)
    if not args.expand_R:
        R = ad.normalize(R, along_dim=0, using_max=False)

    # split the XYR data
    if args.should_test:
        # training and testing, shuffle and split the data
        shuffle_order = np.random.permutation(T)
        trainX, testX = ad.split_along_dim(X[shuffle_order], num_train, dim=0)
        trainY, testY = ad.split_along_dim(Y[shuffle_order], num_train, dim=0)
        trainR, testR = ad.split_along_dim(R[shuffle_order], num_train, dim=0)
        u_train, u_test = ad.split_along_dim(u[shuffle_order], num_train, dim=0)   
    else:
        # no testing, split the data -> test Matrices are empty
        trainX, testX = ad.split_along_dim(X, num_train, dim=0)
        trainY, testY = ad.split_along_dim(Y, num_train, dim=0)
        trainR, testR = ad.split_along_dim(R, num_train, dim=0)
        u_train, u_test = ad.split_along_dim(u, num_train, dim=0)

    # change the input data into pytorch tensors and variables
    trainR, testR = torchten(trainR), torchten(testR)
    u_train, u_test = torchten(u_train), torchten(u_test)
    trainX = Variable(torchten(trainX), requires_grad=False)
    trainY = Variable(torchten(trainY), requires_grad=False)
    testX = Variable(torchten(testX), requires_grad=False)
    testY = Variable(torchten(testY), requires_grad=False)

    if args.expand_R:
        # expand R (trainR and testR)
        trainR_ext = torchten(num_train, J, 15)
        testR_ext = torchten(num_test, J, 15)
        for t in xrange(num_train):
            trainR_ext[t] = expand_R(trainR[t], R_max=15)
        for t in xrange(num_test):
            testR_ext[t] = expand_R(testR[t], R_max=15)
        trainR, testR = trainR_ext, testR_ext
        numFeatures += 14

def make_rand_data(X_max=100.0, R_max=100.0):
    """
    Creates random X and R and calculates Y based on random weights
    """
    # create random X and R and w
    X = ad.normalize(np.floor(np.random.rand(T, J) * X_max), along_dim=1, using_max=False)
    R = torchten(ad.normalize(np.floor(np.random.rand(T, J) * R_max), along_dim=0, using_max=False))
    w = Variable(torch.randn(J, numFeatures, 1).type(torchten))
    Y = np.empty([T, J])
    X, Y = Variable(torchten(X), requires_grad=False), Variable(torchten(Y), requires_grad=False)
    if args.cuda:
        X, Y, R, w = X.cuda(), Y.cuda(), R.cuda(), w.cuda()
    
    # build Y
    for t in xrange(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)
        
        # feed in data
        inp = torchfun.relu(torch.bmm(inp, w)).view(-1, J)
        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        P = torchfun.softmax(inp).t()   # P is now weighted -> softmax
        
        # calculate Y
        Y[t] = torch.mv(P, X[t])
        print(t)

    # for verification of random data, save weights ---------------------------
    w_matrix = w.data.view(-1, numFeatures).cpu().numpy()
    np.savetxt(randXYR_weights_file, w_matrix, fmt="%.15f", delimiter=" ")
    # -------------------------------------------------------------------------

    return (X.data.cpu().numpy(), Y.data.cpu().numpy(), R.cpu().numpy())

def test_given_data(X, Y, R, w, J, T, u):
    loss_normalizer = (torch.mv(torch.t(Y - \
        torch.mean(Y).expand_as(Y)).data, u)).pow(2).sum()
    loss = 0

    for t in xrange(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)
        
        # feed in data
        inp = torchfun.relu(torch.bmm(inp, w)).view(-1, J)
        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        P = torchfun.softmax(inp).t()   # P is now weighted -> softmax
        
        # calculate loss
        Pxt = torch.mv(P, X[t])
        loss += (u[t] * (Y[t] - Pxt)).pow(2).sum()
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    loss /= loss_normalizer
    print("Loss = %f" % loss.data[0])

# =============================================================================
# MyNet class
# =============================================================================
class MyNet(nn.Module):

    def __init__(self):
        """
        Initializes MyNet
        """
        super(MyNet, self).__init__()
        self.w = nn.Parameter(torch.randn(J, numFeatures, 1).type(torchten))

    def forward(self, inp):
        """
        Forward in the network; multiply the weights, take the relu and return
        the softmax
        """    
        # weight -> relu -> softmax
        inp = torchfun.relu(torch.bmm(inp, self.w)).view(-1, J)

        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        # 	 eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        return torchfun.softmax(inp)

# =============================================================================
# training and testing routines
# =============================================================================
def train(net, optimizer, loss_normalizer, u):
    """
    Trains the NN using MyNet
    """
    loss = 0
    P_data = torch.zeros(num_train, J)
    start_time = time.time()
    
    for t in xrange(num_train):
        # build the input by appending trainR[t] to F_DIST
        inp = build_input(trainR[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)
    
        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax
    
        # calculate loss
        Pxt = torch.mv(P, trainX[t])
        P_data[t] = Pxt.data
        loss += (u[t] * (trainY[t] - Pxt)).pow(2).sum()
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    loss /= loss_normalizer
    
    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = time.time()
    return (end_time - start_time, loss.data[0], 
        torch.mean(P_data, dim=0).squeeze().cpu().numpy())

def test(net, loss_normalizer, u):
    """
    Test the network using MyNet
    """
    loss = 0
    P_data = torch.zeros(num_test, J)
    start_time = time.time()
    
    for t in xrange(num_test):
        # build the input by appending testR[t]
        inp = build_input(testR[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)
        
        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax
        
        # calculate loss
        Pxt = torch.mv(P, testX[t])
        P_data[t] = Pxt.data
        loss += (u[t] * (testY[t] - Pxt)).pow(2).sum()
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    loss /= loss_normalizer

    end_time = time.time()
    return (end_time - start_time, loss.data[0], 
        torch.mean(P_data, dim=0).squeeze().cpu().numpy())

# =============================================================================
# utility functions for training and testing routines
# =============================================================================
def build_input(rt):
    """
    Builds the final input for the NN. Joins F_DIST and expanded R
    """
    if args.expand_R:
        return torch.cat([F_DIST, rt.repeat(J, 1, 1)], dim=2)
    return torch.cat([F_DIST, rt.repeat(J, 1)], dim=2)

# =============================================================================
# logs and plots
# =============================================================================
def save_plot(file_name, x, y, xlabel, ylabel, title):
    """
    Saves and shows the loss plot of train and test periods
    """
    # get the losses from data
    train_losses = [i for j in y[0] for i in j][1::2]
    test_losses = [i for j in y[1] for i in j][1::2]
    
    # plot details
    loss_fig = plt.figure(1)
    train_label, = plt.plot(x, train_losses, "r-", label="Train Loss") 
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both", color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
    plt.xlabel(xlabel)

    if args.should_test:
        test_label, = plt.plot(x, test_losses, "b-", label="Test Loss")
        plt.legend(handles=[train_label, test_label])
    else:
        plt.legend(handles=[train_label])
    
    plt.title(title)
    loss_fig.savefig(file_name, bbox_inches="tight", dpi=200)
    if not args.hide_loss_plot:
        plt.show()
    plt.close()

def save_log(file_name, x, y, title):
    """
    Saves the log of train and test periods to a file
    """
    with open(file_name, "wt") as f:
        f.write(title + "\n")
        for i in range(0, len(x), args.log_interval):
            f.write("epoch = %d\t\ttrainloss = %.4f, traintime = %.4f" % (
                x[i], y[0][i][1], y[0][i][0]))
            if args.should_test:
                f.write("\t\ttestloss = %.4f, testtime = %.4f" % (
                    y[1][i][1], y[1][i][0]))
            f.write("\n")

def find_idx_of_nearest_el(array, value):
    """
    Helper function to plot_predicted_map(). Finds the index of the element in
    array closest to value
    """
    return (np.abs(array - value)).argmin()

def plot_predicted_map(file_name, lat_long, point_info, title, plot_offset=0.05):
    """
    Plots the a scatter plot of point_info on the map specified by the lats
    and longs and saves to a file
    """
    # find the dimensions of the plot
    lati = lat_long[:, 0]
    longi = lat_long[:, 1]
    lo_min, lo_max = min(longi) - plot_offset, max(longi) + plot_offset
    la_min, la_max = min(lati) - plot_offset, max(lati) + plot_offset
    plot_width = max(lo_max - lo_min, la_max - la_min)
    lo_max = lo_min + plot_width
    la_max = la_min + plot_width

    lo_range = np.linspace(lo_min, lo_max, num=J + 10, retstep=True)
    la_range = np.linspace(la_min, la_max, num=J + 10, retstep=True)

    lo, la = np.meshgrid(lo_range[0], la_range[0])

    z = np.zeros([J + 10, J + 10])
    for k in xrange(J):
        # find lati[k] in the mesh, longi[k] in the mesh
        lo_k_mesh = find_idx_of_nearest_el(lo[0], longi[k])
        la_k_mesh = find_idx_of_nearest_el(la[:, 0], lati[k])
        z[lo_k_mesh][la_k_mesh] = point_info[k]

    map_fig = plt.figure(2)
    plt.pcolormesh(lo, la, z, cmap=plt.cm.get_cmap('Greys'), vmin=0.0, vmax=0.01)
    plt.axis([lo.min(), lo.max(), la.min(), la.max()])
    plt.colorbar()
    plt.title(title)
    map_fig.savefig(file_name, bbox_inches="tight", dpi=200)
    if not args.hide_map_plot:
        plt.show()
    plt.close()

# =============================================================================
# misc utility functions
# =============================================================================
def expand_R(rt, R_max=15):
    """
    Expands R into a matrix with each R[u] having R_max elements,
    where the first R[u] columns are 1's and rest 0's
    """
    newrt = torchten(J, 15)
    if args.cuda:
        newrt = newrt.cuda()
    for u in xrange(J):
        r = int(rt[u])
        newrt[u] = torch.cat([torch.ones(r), torch.zeros(R_max - r)], dim=0)
    return newrt

# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    # READY!!
    read_set_data()
    net = MyNet()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # SET!!
    if args.cuda:
        # transfer net and tensors to the gpu
        net.cuda()
        trainX, trainY, trainR = trainX.cuda(), trainY.cuda(), trainR.cuda()
        testX, testY, testR = testX.cuda(), testY.cuda(), testR.cuda()
        F_DIST = F_DIST.cuda()
        u_train, u_test = u_train.cuda(), u_test.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "
    if args.expand_R:
        file_pre_gpu = "expandedR, " + file_pre_gpu

    # scalar + tensor currently not supported in pytorch
    train_loss_normalizer = (torch.mv(torch.t(trainY - \
        torch.mean(trainY).expand_as(trainY)).data, u_train)).pow(2).sum()

    if args.should_test:
        test_loss_normalizer = (torch.mv(torch.t(testY - \
            torch.mean(testY).expand_as(testY)).data, u_test)).pow(2).sum()
    
    # GO!!
    train_time_loss, test_time_loss, total_time = [], [], 0.0
    #
    # print("Testing loss before training...")
    # test_given_data(
    #     torch.cat([trainX, testX]),
    #     torch.cat([trainY, testY]),
    #     torch.cat([trainR, testR]),
    #     net.w, J, T)
    #
    for e in xrange(1, args.epochs + 1):
        # train
        train_res = train(net, optimizer, train_loss_normalizer, u_train)
        train_time_loss.append(train_res[0:2])
        total_time += (train_res[0])

        # print results
        if e % 20 == 0:
            print("e= %2d, loss=%.8f" % (e, train_res[1]), end="")

        if args.should_test:
            # test
            test_res = test(net, test_loss_normalizer, u_test)
            test_time_loss.append(test_res[0:2])
            total_time += test_res[0]
            if e % 20 == 0:
                print(", testloss=%.8f\n" % (test_res[1]), end="")
        else:
            print("\n", end="")

        if e == args.epochs:
            # NN's final prediction
            y_pred = test_res[2] if args.should_test else train_res[2]
        
    # FINISH!!
    # log and plot the results: epoch vs loss
    if args.rand:
        file_pre = "randXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
    else:
        file_pre = "origXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)

    log_name = "train=%3.0f%%, lr=%.3e, time=%.4f sec" % (
        args.train_percent * 100, args.lr, total_time)
    
    epoch_data = np.arange(1, args.epochs + 1)
    fname = file_pre_gpu + file_pre + log_name
    
    # save amd plot data
    save_plot("./stats/find_weights/plots/" + fname + ".png", epoch_data, 
        [train_time_loss, test_time_loss], "epoch", "loss", log_name)
    save_log("./stats/find_weights/logs/" + fname + ".txt", epoch_data, 
        [train_time_loss, test_time_loss], log_name)
    plot_predicted_map("./stats/find_weights/map_plots/" + fname + ".png",
            ad.read_lat_long_from_Ffile("./data/loc_feature_with_avicaching_combined.csv", J),
            y_pred, log_name)
    np.savetxt("./stats/find_weights/weights/" + fname + ".txt", 
        net.w.data.view(-1, numFeatures).cpu().numpy(), fmt="%.15f", delimiter=" ")

    print("---> " + fname + " DONE")
