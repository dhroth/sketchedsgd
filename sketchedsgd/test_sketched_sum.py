import torch
import numpy as np
from sketched_optimizer import SketchedSGD, SketchedSum

torch.manual_seed(42)

X = torch.randn(1000, 200)
y = torch.randn(1000, 1)

net = torch.nn.Sequential(torch.nn.Linear(200, 1))
#w = torch.zeros(5, 1)
opt = torch.optim.SGD(net.parameters(), lr=0.0001)
opt = SketchedSGD(net.parameters(), opt,
                  accumulateError=True, k=20, p1=0, p2=4)

sketchedSum = SketchedSum(net, opt, 20, 5, 4)

for i in range(20):
    opt.zero_grad()

    yPred = net(X)
    loss = (yPred - y)**2

    #loss = loss.sum()
    loss = sketchedSum(loss)
    print(loss.item())

    loss.backward()

    opt.step()
