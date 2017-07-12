#!/usr/bin/env python
from __future__ import print_function
import numpy as np, sys, time, argparse
import avicaching_data as ad, lp
import torch, torch.nn as nn, torch.nn.functional as torchfun, torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="NN Avicaching model for finding rewards")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=100, metavar="E",
    help="inputs the number of epochs to train for")

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

torchten = torch.FloatTensor
J, T, totalR, numF = 100, 20, 1000, 10
torch.manual_seed(1)
np.random.seed(seed=1)
if args.cuda:
    torch.cuda.manual_seed(1)

# shapes of datasets -- [] means expanded form:
# - X, Y: J
# - net.R: J [x J x 1]
# - F_DIST: J x J x numF
# - F_DIST_w1: J x J x numF
# - w1: J x J x numF
# - w2: J x numF [x 1]
# - w1_for_r: J x 1 x numF

F_DIST_w1 = torch.randn(J, J, numF)
X, Y = torch.rand(J), torch.rand(J)
loss = 0
w1_for_r = torch.randn(J, 1, numF)
w2 = torch.randn(J, numF, 1)

F_DIST_w1 = Variable(F_DIST_w1, requires_grad=False)
w1_for_r, X = Variable(torchten(w1_for_r), requires_grad=False), Variable(torchten(X), requires_grad=False)
w2 = Variable(torchten(w2), requires_grad=False)

###### ONLY LP
lp_A, lp_c = lp.build_A(J), lp.build_c(J)

for e in xrange(args.epochs):
    r_on_cpu = np.random.randn(J)
    start_lp_time = time.time()
    
    # CONSTRAIN -- LP
    # 1.0 is the sum constraint of rewards
    # the first J outputs are the new rewards
    lp_res = lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0)
    print(time.time() - start_lp_time)
    # net.R.data = torchten(lp_res.x[:J]).unsqueeze(dim=0)

sys.exit()
###########


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        # initiate R
        self.r = np.random.multinomial(totalR, [1 / float(J)] * J, size=1)
        normalizedR = ad.normalize(self.r, using_max=False)
        self.R = nn.Parameter(torchten(normalizedR))

    def forward(self, wt1, wt2):
        # print(self.R.data * 1000)
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

def go_forward(net):
    global w1_for_r, w2, loss
    start_forward_time = time.time()

    # feed in data
    P = net(w1_for_r, w2).t()
    # calculate loss
    Y = torch.mv(P, X)
    loss = torch.norm(Y - torch.mean(Y).expand_as(Y)).pow(2) / J
    
    return time.time() - start_forward_time

def train(net, optimizer):
    global lp_A, lp_c, loss

    # BACKPROPAGATE
    start_backprop_time = time.time()
    
    optimizer.zero_grad()
    loss.backward()         # calculate grad
    optimizer.step()        # update rewards
    
    backprop_time = time.time() - start_backprop_time
    r_on_cpu = net.R.data.squeeze().cpu().numpy()   # transfer data for lp

    start_lp_time = time.time()
    
    # CONSTRAIN -- LP
    # 1.0 is the sum constraint of rewards
    # the first J outputs are the new rewards
    lp_res = lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0)
    lp_time = time.time() - start_lp_time
    print(lp_time)
    net.R.data = torchten(lp_res.x[:J]).unsqueeze(dim=0)

    if args.cuda:
        # transfer data
        net.R.data = net.R.data.cuda()
    
    # FORWARD
    forward_time = go_forward(net)

    return (backprop_time + lp_time + forward_time, lp_time)

# main script
net = MyNet()
if args.cuda:
    net.cuda()
    w1_for_r, w2, F_DIST_w1, X = w1_for_r.cuda(), w2.cuda(), F_DIST_w1.cuda(), X.cuda()

# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
optimizer = optim.Adam(net.parameters(), 0.001)
lp_A, lp_c = lp.build_A(J), lp.build_c(J)

total_lp_time = 0
total_time = go_forward(net)
print(loss.data[0])
for e in xrange(1, args.epochs + 1):
    train_t = train(net, optimizer)
    curr_loss = loss.data[0]

    total_time += train_t[0]
    total_lp_time += train_t[1]
    if e % 20 == 0:
        print("epoch=%5d, loss=%.10f" % (e, curr_loss))

print(total_time, total_lp_time)
