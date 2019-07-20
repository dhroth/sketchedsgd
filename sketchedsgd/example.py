import torch
import numpy as np
from sketched_optimizer import SketchedSGD, SketchedSum, SketchedModel

torch.manual_seed(42)

X = torch.randn(1000, 200)
y = torch.randn(1000, 1)

model = torch.nn.Sequential(torch.nn.Linear(200, 1))
model = SketchedModel(model)

opt = torch.optim.SGD(model.parameters(), lr=0.0001)
opt = SketchedSGD(opt, k=20, accumulateError=True, p1=0, p2=4)

summer = SketchedSum(opt, c=20, r=5, numWorkers=4)

for i in range(20):
    opt.zero_grad()

    yPred = model(X)
    loss = (yPred - y)**2

    loss = summer(loss)

    print("[{}] Loss: {}".format(i, loss.item()))

    loss.backward()

    opt.step()
