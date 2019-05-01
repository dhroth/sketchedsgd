import numpy as np
import torch
from csvec import CSVec
from torch.cuda._utils import _get_device_index
from torch.nn.parallel.scatter_gather import scatter_kwargs, scatter, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn as nn

import ipdb
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
    def __init__(self, opt, k, accumulateError=True, p1=0, p2=0):
        # nesterov not supported
        assert(opt.defaults["nesterov"] == False)
        self.opt = opt
        self.momentum = opt.defaults["momentum"]
        self.weight_decay = opt.defaults["weight_decay"]
        # take the actual steps with basicOpt, since the computation
        # of the weight update is done jointly between the workers
        # and the master in sketchedSum
        #self.basicOpt = torch.optim.SGD(opt.param_groups, lr=1)
        params = []
        for group in opt.param_groups:
            for p in group["params"]:
                params.append(p)
        self.basicOpt = torch.optim.SGD(params, lr=1)
        self.k = k
        self.doAccumulateError = accumulateError
        self.p1 = p1
        self.p2 = p2

    def zero_grad(self):
        self.basicOpt.zero_grad()

    def step(self):
        # the weight update, including lr, momentum, weight decay,
        # and error accumulation, was calculated by sketchedSum
        # and is in self.opt.param_groups
        self.basicOpt.step()

    def step_and_update_lr(self):
        self.step()

    def __getattr__(self, name):
        return getattr(self.opt, name)

    def __setattr__(self, name, value):
        if name == "opt":
            self.__dict__["opt"] = value
        else:
            opt = self.__dict__["opt"]
            setattr(opt, name, value)


class SketchedSum:
    def __init__(self, opt, c, r, numWorkers,
                 numBlocks=1, doTrueTopk=False):
        self.opt = opt
        D = 0
        for group in opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    D += np.prod(p.data.shape)
        self.D = D
        print("D", self.D)
        self.c = c
        self.r = r
        self.numWorkers = numWorkers
        self.doTrueTopk = doTrueTopk
        if opt.param_groups[0]["params"][0].is_cuda:
            self.modelDevice = "cuda"
        else:
            self.modelDevice = "cpu"
        self.device = "cuda"
        print("making sketches")
        print("device", self.device)
        self.us = [torch.zeros(D, device=self.device)
                   for _ in range(numWorkers)]
        self.vs = [torch.zeros(D, device=self.device)
                   for _ in range(numWorkers)]

        if not self.doTrueTopk:
            # don't need sketches for true topk
            self.workerSketches = [CSVec(d=D, c=c, r=r,
                                         device=self.device, nChunks=1,
                                         numBlocks=numBlocks)
                                   for _ in range(numWorkers)]

    def _getGradShapes(self):
        with torch.no_grad():
            gradShapes = []
            gradSizes = []
            for group in self.opt.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        gradShapes.append(p.data.shape)
                        gradSizes.append(np.prod(p.data.shape))
                    else:
                        gradShapes.append(p.grad.data.shape)
                        gradSizes.append(np.prod(p.grad.data.shape))
            return gradShapes, gradSizes

    def _getGradVec(self):
        gradVec = []
        with torch.no_grad():
            # flatten
            for group in self.opt.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        gradVec.append(torch.zeros_like(p.data.view(-1)))
                    else:
                        gradVec.append(p.grad.data.view(-1).float())

            # concat into a single vector
            gradVec = torch.cat(gradVec)

        return gradVec

    def _getLRVec(self):
        lrVec = []
        for group in self.opt.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    lrVec.append(torch.zeros_like(p.data.view(-1)))
                else:
                    grad = p.grad.data.view(-1)
                    lrVec.append(torch.ones_like(grad) * lr)
        return torch.cat(lrVec)

    def _getParamVec(self):
        d = []
        for group in self.opt.param_groups:
            for p in group["params"]:
                d.append(p.data.view(-1).float())
        return torch.cat(d).to(self.device)

    def _setGradVec(self, vec):
        # put vec into p.grad.data
        vec = vec.to(self.modelDevice)
        gradShapes, gradSizes = self._getGradShapes()
        startPos = 0
        i = 0
        for group in self.opt.param_groups:
            for p in group["params"]:
                shape = gradShapes[i]
                size = gradSizes[i]
                i += 1
                if p.grad is None:
                    continue

                assert(size == np.prod(p.grad.data.size()))
                p.grad.data.zero_()
                p.grad.data.add_(vec[startPos:startPos + size].reshape(shape))
                startPos += size

    def print_graph(self, g, level=0):
        # just for debugging
        if g is None: return
        print('*'*level, g)
        for subg in g.next_functions:
            self.print_graph(subg[0], level+1)

    def __call__(self, loss):
        self.loss = loss
        batchSize = loss.size()[0]
        self.losses = []
        for i in range(self.numWorkers):
            start = i * batchSize // self.numWorkers
            end = (i + 1) * batchSize // self.numWorkers
            self.losses.append(loss[start:end].sum() / self.numWorkers)
        return self

    def _backwardWorker(self, workerId, doAggregate=True):
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
        lrVec = self._getLRVec()
        #print("LR:", lrVec)
        gradVec *= lrVec

        if self.opt.doAccumulateError:
            self.us[workerId].mul_(self.opt.momentum).add_(gradVec)
            self.vs[workerId] += self.us[workerId]
        else:
            self.vs[workerId] += gradVec

        # doAggregate means we're going to aggregate all the workers
        # after this gradient computation step (otherwise, we plan
        # to aggregate additional gradients before aggregating on
        # the parameter server)
        if doAggregate and not self.doTrueTopk:
            # sketch the current (modified) gradient in preparation for
            # aggregation by the parameter server
            self.workerSketches[workerId].zero()
            if self.opt.doAccumulateError:
                # sketch vs[workerId] into self.workerSketches[workerId]
                if self.opt.p1 > 0:
                    # truncate and then sketch
                    tk = topk(self.vs[workerId], self.opt.p1 * self.opt.k)
                    self.workerSketches[workerId] += tk
                else:
                    # sketch the full vector
                    self.workerSketches[workerId] += self.vs[workerId]
            else:
                # if no error accumulation, then self.vs just accumulates
                # gradients directly until we're ready to aggregate
                self.workerSketches[workerId] += self.vs[workerId]

    def _aggregateSketches(self):
        weightUpdate = None
        if self.opt.doAccumulateError:
            # get candidate topk, then do second round of communication
            if self.opt.p2 > 0:
                candidateTopk = np.sum(self.workerSketches).unSketch(
                                    k=self.opt.p2*self.opt.k)
                # get coords that were populated by the unSketch
                # (i.e. the heavy hitters)
                candidateHHCoords = candidateTopk.nonzero()
                # get exact values for candidateHHCoords
                candidateTopk[candidateHHCoords] = torch.sum(torch.cat(
                        [self.vs[workerId][candidateHHCoords]
                         for workerId in range(self.numWorkers)],
                    dim=1),
                dim=1)[:,np.newaxis]
                weightUpdate = topk(candidateTopk, k=self.opt.k)
                #weightUpdate = topk(sum(self.vs), k=self.opt.k)
            else:
                # if p2 == 0, then there's no second round of
                # communication: we just use the values for the gradient
                # that we got from the sketch
                assert(self.opt.p2 == 0)
                weightUpdate = np.sum(self.workerSketches).unSketch(k=self.opt.k)

            if False:
                # just for debugging
                trueWeightUpdate = topk(sum(self.vs), k=self.opt.k)
                overlap = torch.sum((weightUpdate != 0) * (trueWeightUpdate != 0)).item()
                print("OVERLAP:", overlap, "out of ", self.opt.k)
                if True or overlap < 7000:
                    ipdb.set_trace()
                print("(nonzero WU):", weightUpdate.nonzero().size())
        else:
            # no error accumulation -- gradVecs were sketched directly
            weightUpdate = np.sum(self.workerSketches).unSketch(k=self.opt.k)
        assert(weightUpdate is not None)
        return weightUpdate

    def _aggregateVs(self):
        return topk(sum(self.vs), k=self.opt.k)

    #@profile
    def backward(self, doAggregate=True):
        # need to save the existing gradient so we can accumulate the
        # new gradient instead of replacing the old
        initialGradVec = self._getGradVec()

        # backprop on each worker updating self.us and self.vs
        for workerId in range(self.numWorkers):
            # if doAggregate, _backwardWorker will sketch self.vs[workerId]
            # into self.workerSketches, so that self._aggregateSketches
            # can aggregate them into the final weight update
            self._backwardWorker(workerId, doAggregate)

        if doAggregate:
            if self.doTrueTopk:
                # for true top-k, just aggregate self.vs directly
                weightUpdate = self._aggregateVs()
                #print(torch.norm(weightUpdate))
            else:
                # for sketched top-k, aggregate the sketches
                weightUpdate = self._aggregateSketches()
                #print(torch.norm(weightUpdate))

            if self.opt.doAccumulateError:
                # zero out coordinates on each worker that the parameter
                # server updates
                hhCoords = weightUpdate.nonzero()
                #print("HH nonzero", hhCoords.size())
                for workerId in range(self.numWorkers):
                    self.us[workerId][hhCoords] = 0
                    self.vs[workerId][hhCoords] = 0
            else:
                # if no error accumulation, self.vs just accumulates
                # gradients directly until we aggregate them, at which
                # point each worker is completely zeroed out
                for workerId in range(self.numWorkers):
                    self.vs[workerId].zero_()

            # add back the initial gradient vector
            weightUpdate.add_(initialGradVec)

            self._setGradVec(weightUpdate)
        else:
            # if we're not aggregating, then put back the initialGradVec
            # (since self._backwardWorker may have modified it)
            self._setGradVec(initialGradVec)


    def item(self):
        with torch.no_grad():
            return self.loss.sum().item()

    def __div__(self, factor):
        return self.div(factor)
    def __truediv__(self, factor):
        return self.div(factor)

    def __mul__(self, factor):
        return self.mul(factor)

    def div(self, factor):
        assert(self.loss is not None)
        self.loss = self.loss / factor
        for i in range(self.numWorkers):
            self.losses[i] = self.losses[i] / factor
        return self

    def mul(self, factor):
        assert(self.loss is not None)
        self.loss = self.loss * factor
        for i in range(self.numWorkers):
            self.losses[i] = self.losses[i] * factor
        return self
