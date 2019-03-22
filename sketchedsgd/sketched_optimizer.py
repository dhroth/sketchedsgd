import numpy as np
import torch
from csvec import CSVec
from torch.cuda._utils import _get_device_index
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn as nn

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

def topk(vec, k):
    ret = torch.zeros_like(vec)
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)
    ret[topkIndices] = vec[topkIndices]
    return ret


class SketchedSGD(torch.optim.Optimizer):
    def __init__(self, params, opt, k, accumulateError=True, p1=0, p2=0):
        # TODO probably can eliminate need for params arg by inspecting
        # opt

        # nesterov not supported
        assert(opt.defaults["nesterov"] == False)
        self.opt = opt
        self.momentum = opt.defaults["momentum"]
        self.weight_decay = opt.defaults["weight_decay"]
        # take the actual steps with basicOpt, since the computation
        # of the weight update is done jointly between the workers
        # and the master in sketchedSum
        #self.basicOpt = torch.optim.SGD(opt.param_groups, lr=1)
        self.basicOpt = torch.optim.SGD(params, lr=1)
        self.k = k
        self.doAccumulateError = accumulateError
        self.p1 = p1
        self.p2 = p2

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        # the weight update, including lr, momentum, weight decay,
        # and error accumulation, was calculated by sketchedSum
        # and is in self.opt.param_groups
        self.basicOpt.step()

    def step_and_update_lr(self):
        self.step()

class SketchedModel(nn.DataParallel):
    def __init__(self, model, numWorkers,
                 devices=None, outputDevice=None, dim=0):
        super().__init__(model, devices, outputDevice, dim)
        """
        self.model = model
        self.numWorkers = numWorkers

        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        if output_device is None:
            outputDevice = devices[0]

        self.devices = list(map(
            lambda x: _get_device_index(x, True), devices
        ))
        self.outputDevice = outputDevice
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])
        """

    def forward(self, *inputs, **kwargs):
        loss = super().forward(*inputs, **kwargs)
        print(loss.size())
        exit()
        # split the loss 
        """
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.devices) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
        """


class SketchedSum:
    def __init__(self, opt, c, r, numWorkers):
        self.opt = opt
        D = 0
        for group in opt.param_groups:
            for p in group:
                D += np.prod(p.shape)
        self.D = D
        print("D", self.D)
        self.c = c
        self.r = r
        self.numWorkers = numWorkers
        if params[0].is_cuda:
            self.modelDevice = "cuda"
        else:
            self.modelDevice = "cpu"
        self.device = "cuda"
        print("making sketches")
        print("device", self.device)
        self.workerSketches = [CSVec(d=D, c=c, r=r,
                                     device=self.device, nChunks=4)
                               for _ in range(numWorkers)]
        self.us = [torch.zeros(D, device=self.device)
                   for _ in range(numWorkers)]
        self.vs = [torch.zeros(D, device=self.device)
                   for _ in range(numWorkers)]

    def _getGradShapes(self):
        with torch.no_grad():
            gradShapes = []
            gradSizes = []
            for group in self.opt.param_groups:
                for p in group:
                    if p.grad is not None:
                        gradShapes.append(p.grad.data.shape)
                        gradSizes.append(np.prod(p.grad.data.shape))
            return gradShapes, gradSizes

    def _getGradVec(self):
        with torch.no_grad():
            # flatten
            gradVec = []
            for group in self.opt.param_groups:
                for p in group:
                    if p.grad is not None:
                        gradVec.append(p.grad.data.view(-1).float())

            # concat into a single vector
            gradVec = torch.cat(gradVec)

            return gradVec

    def _getLRVec(self):
        lrVec = []
        for group in self.opt.param_groups:
            for p in group:
                if p.grad is not None:
                    lrVec.append(group["lr"])

    def _getParamVec(self):
        d = []
        for group in self.opt.param_groups:
            for p in group:
                d.append(p.data.view(-1).float())
        return torch.cat(d).to(self.device)

    def _setGradVec(self, vec):
        # put vec into p.grad.data
        vec = vec.to(self.modelDevice)
        gradShapes, gradSizes = self._getGradShapes()
        startPos = 0
        for group, shape, size in zip(self.opt.param_groups, gradShapes, gradSizes):
            for p in group:
                if p.grad is None:
                    continue

                assert(size == np.prod(p.grad.data.size()))
                p.grad.data.zero_()
                p.grad.data.add_(vec[startPos:startPos + size].reshape(shape))
                startPos += size

    def __call__(self, loss):
        self.loss = loss
        batchSize = loss.size()[0]
        self.losses = []
        for i in range(self.numWorkers):
            start = i * batchSize // self.numWorkers
            end = (i + 1) * batchSize // self.numWorkers
            self.losses.append(loss[start:end].sum() / self.numWorkers)
        return self

    #@profile
    def backward(self):
        #grads = []
        for workerId in range(self.numWorkers):
            if workerId == self.numWorkers - 1:
                retain_graph = False
            else:
                retain_graph = True
            self.opt.zero_grad()
            self.losses[workerId].backward(retain_graph=retain_graph)
            gradVec = self._getGradVec().to(self.device)
            # do weight decay right away
            # divide by num_workers because the gradient is
            # summed on master instead of averaged (and the
            # loss above is divided by num_workers)
            if self.opt.weight_decay != 0:
                gradVec.add_(self.opt.weight_decay / self.numWorkers,
                             self._getParamVec())
            # multiply by learning rate before doing momentum
            # & error accumulation
            gradVec *= self._getLRVec()
            # sketch and send the current (modified) gradient
            self.workerSketches[workerId].zero()
            if self.opt.doAccumulateError:
                # sketch and send vs[workerId]
                #self.us[workerId] = self.opt.momentum * self.us[workerId] + gradVec
                self.us[workerId].mul_(self.opt.momentum).add_(gradVec)
                self.vs[workerId] += self.us[workerId]
                if self.opt.p1 > 0:
                    # truncate and then sketch
                    tk = topk(self.vs[workerId], self.opt.p1 * self.opt.k)
                    self.workerSketches[workerId] += tk
                else:
                    # sketch the full vector
                    self.workerSketches[workerId] += self.vs[workerId]
            else:
                # no error accumulation, so sketch and send gradVec
                self.workerSketches[workerId] += gradVec
            #grads.append(gradVec)
        
        # code here is run on the "master" node
        if self.opt.doAccumulateError:
            # get candidate topk, then do second round of communication
            if self.opt.p2 > 0:
                candidateTopk = np.sum(self.workerSketches).unSketch(
                                    k=self.opt.p2*self.opt.k)
                # get coords that were populated by the unSketch
                # (i.e. the heavy hitters)
                candidateHHCoords = candidateTopk.nonzero()
                # get exact values for candidateHHCoords
                #tmp = [self.vs[workerId][candidateHHCoords]
                #         for workerId in range(self.numWorkers)]
                #print([t.size() for t in tmp])
                #print("HELLO THERE?", tmp)
                #print(tmp[0] + tmp[1] + tmp[2] + tmp[3])
                candidateTopk[candidateHHCoords] = torch.sum(torch.cat(
                        [self.vs[workerId][candidateHHCoords]
                         for workerId in range(self.numWorkers)],
                    dim=1),
                dim=1)[:,np.newaxis]
                weightUpdate = topk(candidateTopk, k=self.opt.k)
            else:
                assert(self.opt.p2 == 0)
                weightUpdate = np.sum(self.workerSketches).unSketch(k=self.opt.k)

            hhCoords = weightUpdate.nonzero()
            for workerId in range(self.numWorkers):
                self.us[workerId][hhCoords] = 0
                self.vs[workerId][hhCoords] = 0
        else:
            # no error accumulation -- gradVecs were sketched directly
            weightUpdate = np.sum(self.workerSketches).unSketch(k=self.opt.k)

        self._setGradVec(weightUpdate)

    def item(self):
        with torch.no_grad():
            return self.loss.sum().item()
