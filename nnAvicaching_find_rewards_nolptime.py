#!/usr/bin/env python
from __future__ import print_function
import torch, torch.nn as nn, torch.nn.functional as torchfun, torch.optim as optim
from torch.autograd import Variable
import numpy as np, argparse, time, os, sys
import avicaching_data as ad, lp, matplotlib
try:
    os.environ["DISPLAY"]
except KeyError as e:
    # working without X/GUI environment
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# options
# =============================================================================
parser = argparse.ArgumentParser(description="NN Avicaching model for finding rewards")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
    help="inputs learning rate of the network (default=0.01)")
parser.add_argument("--momentum", type=float, default=1.0, metavar="M",
    help="inputs SGD momentum (default=1.0)")
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
parser.add_argument("--rewards", type=float, default=1000.0, metavar="R",
    help="inputs the total budget of rewards to be distributed (default=1000.0)")
parser.add_argument("--rand", action="store_true", default=False,
    help="uses random data")
parser.add_argument("--weights-file", type=str, 
    default="./stats/find_weights/weights/gpu, origXYR_seed=1, epochs=10000, train= 80%, lr=1.000e-02, time=1260.1639 sec.txt", 
    metavar="f", help="inputs the location of the file to use weights from")
parser.add_argument("--test", type=str, default="", 
    metavar="t", help="inputs the location of the file to test rewards from")
parser.add_argument("--log-interval", type=int, default=1, metavar="I",
    help="prints training information at I epoch intervals (default=1)")
parser.add_argument('--seed', type=int, default=1, metavar='S',
    help='seed (default=1)')

parser.add_argument("--hide-loss-plot", action="store_true", default=False,
    help="hides the loss plot, which is only saved")

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =============================================================================
# parameters and constants
# =============================================================================
J, T, weights_file_name, totalR = args.locations, args.time, args.weights_file, args.rewards
X, w1_for_r, w2, F_DIST_w1, numFeatures = [], [], [], [], 0
torchten = torch.FloatTensor
lp_A, lp_c = [], []

randXYR_file = "./data/random/randXYR" + str(116) + ".txt"
randF_file = "./data/random/randF" + str(116) + ".csv"
randDIST_file = "./data/random/randDIST" + str(116) + ".txt"

# =============================================================================
# data input
# =============================================================================
def read_set_data():
    global X, numFeatures, F_DIST_w1, w1_for_r, w2
    # read f and dist datasets from file, operate on them
    if args.rand:
        F = ad.read_F_file(randF_file, J)
        DIST = ad.read_dist_file(randDIST_file, J)
    else:
        F = ad.read_F_file("./data/loc_feature_with_avicaching_combined.csv", J)
        DIST = ad.read_dist_file("./data/site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

    # process data for the NN
    F, DIST = ad.normalize(F, along_dim=0, using_max=True), ad.normalize(DIST, using_max=True)  # normalize using max
    numFeatures = len(F[0]) + 1     # distance included
    F_DIST = torchten(ad.combine_DIST_F(F, DIST, J, numFeatures))
    numFeatures += 1                # for rewards later

    # read W and X
    w1, w2 = ad.read_weights_file(weights_file_name, J, numFeatures)
    w2 = np.expand_dims(w2, axis=2)

    if args.rand:
        X, _, _ = ad.read_XYR_file(randXYR_file, J, T)
    else:
        X, _, _ = ad.read_XYR_file("./data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)
    
    # split w1; multiply the fdist portion of w1 with F_DIST
    w1_for_fdist, w1_for_r = ad.split_along_dim(w1, numFeatures - 1, dim=1)
    F_DIST_w1 = Variable(torch.bmm(F_DIST, torchten(w1_for_fdist)).squeeze(dim=2), requires_grad=False)

    # condense X along T into a single vector and normalize
    X = ad.normalize(X.sum(axis=0), using_max=False)
    
    w1_for_r, X = Variable(torchten(w1_for_r), requires_grad=False), Variable(torchten(X), requires_grad=False)
    w2 = Variable(torchten(w2), requires_grad=False)
    # w1_for_r shape J x 1 x numF
    # w2 shape J x numF x 1

# =============================================================================
# MyNet class
# =============================================================================
class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        # initiate R
        self.r = np.random.multinomial(totalR, [1 / float(J)] * J, size=1)
        normalizedR = ad.normalize(self.r, using_max=False)
        self.R = nn.Parameter(torchten(normalizedR))

    def forward(self, wt1, wt2):
        repeatedR = self.R.repeat(J, 1).unsqueeze(dim=2)    # shape is J x J x 1
        res = torch.bmm(repeatedR, wt1) + F_DIST_w1     # inp is J x J x numF after
        res = torchfun.relu(res)
        res = torch.bmm(res, wt2).view(-1, J)    # inp is J x J
        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        return torchfun.softmax(res)

def train(net, optimizer):
    global w1_for_r, w2, lp_A, lp_c
    start_train = time.time()

    # feed in data
    P = net(w1_for_r, w2).t()    # P is now weighted -> softmax
    
    # calculate loss
    Y = torch.mv(P, X)
    loss = torch.norm(Y - torch.mean(Y).expand_as(Y)).pow(2) / J

    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    
    # update the rewards and constrain them
    optimizer.step()
    end_train = time.time() - start_train

    r_on_cpu = net.R.data.squeeze().cpu().numpy()   # transfer data for lp
    start_lp = time.time()
    # 1.0 is the sum constraint of rewards
    # the first J outputs are the new rewards
    net.R.data = torchten(lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0).x[:J]).unsqueeze(dim=0)
    end_lp = time.time() - start_lp
    if args.cuda:
        # transfer data
        net.R.data = net.R.data.cuda()

    return (end_train, loss.data[0], net.R.data.sum())

def test_rewards(r):
    global w1_for_r, w2
    start_time = time.time()

    # feed in data
    repeatedR = r.repeat(J, 1).unsqueeze(dim=2)
    inp = torch.bmm(repeatedR, w1_for_r) + F_DIST_w1
    inp = torchfun.relu(inp)
    inp = torch.bmm(inp, w2).view(-1, J)
    # add eta to inp[u][u]
    # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
    # inp += eta_matrix
    P = torchfun.softmax(inp)
    
    # calculate loss
    Y = torch.mv(P, X)
    loss = torch.norm(Y - torch.mean(Y).expand_as(Y)).pow(2) / J

    end_time = time.time()
    return (end_time - start_time, loss.data[0])

# =============================================================================
# logs and plots
# =============================================================================
def save_log(file_name, results, title, rewards=None):
    """
    Saves the log of train and test periods to a file
    """
    with open(file_name, "wt") as f:
        f.write(title + "\n")
        if rewards is not None:
            np.savetxt(f, rewards, fmt="%.15f", delimiter=" ")
        f.write("time = %.4f\t\tloss = %.15f\n" % (results[0], results[1]))

def save_plot(file_name, x, y, xlabel, ylabel, title):
    """
    Saves and shows the loss plot of train and test periods
    """
    # get the losses from data
    train_losses = [i for j in y for i in j][1::2]
    
    # plot details
    loss_fig = plt.figure(1)
    train_label, = plt.plot(x, train_losses, "r-", label="Train Loss") 
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both", color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
    plt.xlabel(xlabel)

    plt.title(title)
    loss_fig.savefig(file_name, bbox_inches="tight", dpi=200)
    if not args.hide_loss_plot:
        plt.show()
    plt.close()

# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    read_set_data()
    if args.test:
        rewards = np.loadtxt(args.test, delimiter=" ")[:J]
        rewards = Variable(torchten(ad.normalize(rewards, using_max=False)))
        res = test_rewards(rewards)
        # save results
        fname = "testing \"" + args.test[args.test.rfind("/") + 1:] + '"'
        save_log("./stats/find_rewards/" + fname + ".txt", res, weights_file_name)
        sys.exit(0)
    
    net = MyNet()
    if args.cuda:
        net.cuda()
        w1_for_r, w2, F_DIST_w1, X = w1_for_r.cuda(), w2.cuda(), F_DIST_w1.cuda(), X.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lp_A, lp_c = lp.build_A(J), lp.build_c(J)

    total_time, best_loss = 0.0, float("inf")
    train_time_loss = []
    for e in xrange(1, args.epochs + 1):
        train_res = train(net, optimizer)
        train_time_loss.append(train_res[0:2])

        if train_res[1] < best_loss:
            # save the best result uptil now
            best_loss = train_res[1]
            best_rew = net.R.data
        
        total_time += train_res[0]
        if e % 20 == 0:
            print("epoch=%5d, loss=%.10f, budget=%.10f" % (e, train_res[1], train_res[2]))

    # save and plot
    best_rew = best_rew.cpu().numpy() * totalR

    if args.rand:
        file_pre = "randXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
    else:
        file_pre = "origXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
    log_name = "lr=%.3e, bestloss=%.6f, time=%.4f sec" % (
        args.lr, best_loss, total_time)
    epoch_data = np.arange(1, args.epochs + 1)
    fname = file_pre_gpu + file_pre + log_name

    save_plot("./stats/find_rewards/plots/" + fname + ".png", epoch_data, 
        train_time_loss, "epoch", "loss", log_name)
    save_log("./stats/find_rewards/logs/" + fname + ".txt", (total_time, best_loss),
        weights_file_name, best_rew)

    print("---> " + fname + " DONE")