import numpy as np
import torch
from csvec import CSVec
from torch.cuda._utils import _get_device_index
from torch.nn.parallel.scatter_gather import scatter_kwargs, scatter, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn as nn
import random

#import ipdb
#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)

    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)

    ret[topkIndices] = vec[topkIndices]
    return ret

def printMemoryUsage():
    import gc
    bigs = []
    totalBytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                if isinstance(obj, torch.cuda.ByteTensor):
                    dsize = 1
                elif isinstance(obj, torch.cuda.FloatTensor) or isinstance(obj, torch.cuda.IntTensor):
                    dsize = 4
                elif isinstance(obj, torch.cuda.DoubleTensor) or isinstance(obj, torch.cuda.LongTensor):
                    dsize = 8
                totalBytes += np.product(obj.size()) * dsize
                if obj.size()[0] > 90000000:
                    bigs.append(obj)
        except:
            pass
    for big in bigs:
        print(big)
    print("Total Size: {} MB".format(totalBytes / 1024 / 1024))


class SketchedSGD(torch.optim.Optimizer):
    """SketchedSGD optimizer

    This is a thin wrapper over optim.SGD. Most of the work to do
    sketching is in SketchedSum. SketchedSum handles the learning rate,
    momentum, and weight decay, so we don't want the user's optim.SGD
    instance to apply them a second time.
    """
    def __init__(self, opt, k, accumulateError=True, p1=0, p2=0):
        """SketchedSGD Constructor

        Args:
            opt: the optim.SGD instance you were using before applying
                 sketching
            k: how many gradient elements to extract from the sketches
            accumulateError: whether or not to accumulate error in the
                             workers
            p1: truncate worker gradients to p1*k before sketching. If
                zero, don't truncate
            p2: the parameter server extracts p2*k heavy hitters from
                the summed sketches, requests p2*k actual gradient values
                from each worker, and then computes the topk of the sum
                of the actual values
        """
        # nesterov not supported
        assert(opt.defaults["nesterov"] == False)
        self.opt = opt
        self.momentum = opt.defaults["momentum"]
        self.weight_decay = opt.defaults["weight_decay"]
        # take the actual steps with basicOpt, since the computation
        # of the weight update is done jointly between the workers
        # and the master in SketchedSum

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
        """Zero out the gradient"""
        self.basicOpt.zero_grad()

    def step(self):
        """Step the optimizer"""
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

class SketchedModel:
    # not inheriting from nn.Module to avoid the fact that implementing
    # __getattr__ on a nn.Module is tricky, since self.model = model
    # doesn't actually add "model" to self.__dict__ -- instead, nn.Module
    # creates a key/value pair in some internal dictionary that keeps
    # track of submodules
    def __init__(self, model, sketchBiases=False):
        self.model = model
        # sketch everything by default
        for p in model.parameters():
            p.do_sketching = True

        if not sketchBiases:
            for m in model.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.do_sketching = True
                    m.bias.do_sketching = False

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)


class SketchedSum:
    """Sums a tensor s.t. gradients of the sum are sketched during backward

    Normally, the loss is computed as
    loss = criterion(predictions, ground_truth).sum()
    where the sum() is over the batch dimension.

    In order to sketch the gradients of loss during the backward()
    computation, replace the above with
    summer = SketchedSum(...)
    loss = summer(criterion(predictions, ground_truth))

    Now, when loss.backward() is called, the gradients in each leaf of
    the computation graph will be the result of computing the gradient
    on several workers, sketching the gradients, summing the sketches,
    and extracting the topk values of the summed sketch, possibly with a
    second round of communication between the workers and parameter server.
    """
    def __init__(self, opt, c, r, numWorkers, numBlocks=1,
                 doTrueTopk=False, doLocalTopk=False, doRandomK=False):
        """SketchedSum constructor

        Args:
            opt: an instance of torch.optim.SGD whose momentum and weight
                 decay we want to emulate
            c: number of columns in the sketch
            r: numbers of rows in the sketch
            numWorkers: how many workers to divide the gradient
                        computation among
            numBlocks: memory optimization for the sketch (higher means
                       less memory used, but randomness becomes correlated)
            doTrueTopk: instead of sketching, compute the true topk
                        of the sum of the workers' gradients
            doLocalTopk: instead of sketching, send and then sum the local
                         topk of each worker's v vector
            doRandomK: instead of sketching, send a random set of
                       k coordinates
        """
        self.opt = opt
        self.c = c
        self.r = r
        self.numWorkers = numWorkers
        # at most one of true topk, local topk, and random k allowed
        # (what can I say -- I don't believe in implicit casting?)
        assert(((1 if doTrueTopk else 0) +
                (1 if doLocalTopk else 0) +
                (1 if doRandomK else 0)) <= 1)
        self.doTrueTopk = doTrueTopk
        self.doLocalTopk = doLocalTopk
        self.doRandomK = doRandomK
        self.doSketching = not (doTrueTopk or doLocalTopk or doRandomK)
        # self.modelDevice is not tested... not sure what happens if
        # the model is on the CPU
        if opt.param_groups[0]["params"][0].is_cuda:
            self.modelDevice = "cuda"
        else:
            self.modelDevice = "cpu"
        self.device = "cuda"
        print("device", self.device)

        D = 0
        sketchMask = []
        for group in opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = np.prod(p.data.shape)
                    if p.do_sketching:
                        sketchMask.append(torch.ones(size))
                    else:
                        sketchMask.append(torch.zeros(size))
                    D += size
        self.D = D
        # a mask indicating which gradient elements we should sketch
        # and which we should send without compression (e.g. bias terms,
        # maybe early layers, etc.)
        self.sketchMask = torch.cat(sketchMask).byte().to(self.device)

        print("D: {}".format(D))
        print("sketchMask.sum(): {}".format(self.sketchMask.sum()))


        self.us = [torch.zeros(D, device=self.device)
                   for _ in range(numWorkers)]
        self.vs = [torch.zeros(D, device=self.device)
                   for _ in range(numWorkers)]

        # don't need sketches for true/local/random topk
        if self.doSketching:
            print("making sketches")
            # dimensionality of the sketch (d) is the number of gradient
            # elements that we're going to sketch, i.e. sketchMask.sum()
            self.workerSketches = [CSVec(d=self.sketchMask.sum().item(),
                                         c=c, r=r,
                                         device=self.device, nChunks=1,
                                         numBlocks=numBlocks)
                                   for _ in range(numWorkers)]
        else:
            print("not making sketches")

    def _getGradShapes(self):
        """Return the shapes and sizes of the weight matrices"""
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
        """Return the gradient flattened to a vector"""
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
        """Return a vector of each gradient element's learning rate

        If all parameters have the same learning rate, this just
        returns torch.ones(D) * learning_rate. In this case, this
        function is memory-optimized by returning just a single
        number.
        """
        if len(self.opt.param_groups) == 1:
            return self.opt.param_groups[0]["lr"]

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
        """Returns the current model weights as a vector"""
        d = []
        for group in self.opt.param_groups:
            for p in group["params"]:
                d.append(p.data.view(-1).float())
        return torch.cat(d).to(self.device)

    def _setGradVec(self, vec):
        """Set the gradient to vec"""
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
        """Partition the loss into numWorkers parts along the batch axis"""
        self.loss = loss
        batchSize = loss.size()[0]
        self.losses = []
        for i in range(self.numWorkers):
            start = i * batchSize // self.numWorkers
            end = (i + 1) * batchSize // self.numWorkers
            self.losses.append(loss[start:end].sum() / self.numWorkers)
        return self

    def _backwardWorker(self, workerId, doAggregate=True):
        """Do a backward pass for one worker

        Args:
            workerId: which worker to do the backward pass for (between
                      0 and self.numWorkers - 1)
            doAggregate: whether or not the next step is to aggregate
                         the workers' gradients. If so, we will sketch
                         the computed gradient. Otherwise, we plan to
                         call backwardWorker() again and accumulate.
                         (bad abstraction and not really tested, sorry)
        """
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
        if doAggregate and self.doSketching:
            # sketch the current (modified) gradient in preparation for
            # aggregation by the parameter server
            self.workerSketches[workerId].zero()
            maskedV = self.vs[workerId][self.sketchMask]
            if self.opt.doAccumulateError:
                # sketch vs[workerId] into self.workerSketches[workerId]
                if self.opt.p1 > 0:
                    # truncate and then sketch
                    tk = topk(maskedV, self.opt.p1 * self.opt.k)
                    self.workerSketches[workerId] += tk
                else:
                    # sketch the full vector
                    self.workerSketches[workerId] += maskedV
            else:
                # if no error accumulation, then self.vs just accumulates
                # gradients directly until we're ready to aggregate
                # TODO doAccumulateError=False not tested, probably broken
                self.workerSketches[workerId] += maskedV

    def _aggregateSketches(self):
        """Aggregate the sketches of each worker

        If p2 > 0, do a second round of communication between the
        parameter server and the workers in order to get a better
        estimate of the topk (both which elements are in the topk and
        the values of those elements)
        """
        weightUpdate = torch.zeros_like(self.vs[0])
        # for coords that we're not sketching, store the sum across
        # workers directly in weightUpdate
        nonSketchedVs = torch.stack(
                            [self.vs[workerId][~self.sketchMask]
                             for workerId in range(self.numWorkers)]
                        )
        weightUpdate[~self.sketchMask] = torch.sum(nonSketchedVs,
                                                   dim=0)
        if self.opt.doAccumulateError:
            # get candidate topk, then do second round of communication
            if self.opt.p2 > 0:
                candidateTopk = np.sum(self.workerSketches).unSketch(
                                    k=self.opt.p2*self.opt.k)
                # get coords that were populated by the unSketch
                # (i.e. the heavy hitters)
                candidateHHCoords = candidateTopk.nonzero()
                # get exact values for candidateHHCoords
                candidateTopk[candidateHHCoords] = torch.sum(torch.stack(
                    [self.vs[workerId][self.sketchMask][candidateHHCoords]
                     for workerId in range(self.numWorkers)]
                    ),
                dim=0)
                weightUpdate[self.sketchMask] = topk(candidateTopk,
                                                     k=self.opt.k)
                #weightUpdate[self.sketchMask = topk(sum(self.vs),
                #                                    k=self.opt.k)
            else:
                # if p2 == 0, then there's no second round of
                # communication: we just use the values for the gradient
                # that we got from the sketch
                assert(self.opt.p2 == 0)
                weightUpdate[self.sketchMask] = np.sum(
                        self.workerSketches).unSketch(k=self.opt.k)

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
        """Aggregate the error accumulation vectors directly

        Used when doing the true topk instead of sketching.
        """
        weightUpdate = torch.zeros_like(self.vs[0])

        # for coords we're not compressing, store the sum of vs
        nonSketchedVs = torch.stack(
                            [self.vs[workerId][~self.sketchMask]
                             for workerId in range(self.numWorkers)]
                        )
        weightUpdate[~self.sketchMask] = torch.sum(nonSketchedVs, dim=0)

        # for coords we are compressing, store the topk of the sum of vs
        sketchedVs = torch.stack(
                        [self.vs[workerId][self.sketchMask]
                         for workerId in range(self.numWorkers)]
                     )
        weightUpdate[self.sketchMask] = topk(torch.sum(sketchedVs, dim=0),
                                             k=self.opt.k)

        return weightUpdate

    def _aggregateTopkVs(self):
        assert(self.opt.p2 in [0, 1])

        """Aggregate the local topk of each workers' v vector"""
        weightUpdate = torch.zeros_like(self.vs[0])

        # for coords we're not compressing, store the sum of vs
        nonSketchedVs = torch.stack(
                            [self.vs[workerId][~self.sketchMask]
                             for workerId in range(self.numWorkers)]
                        )
        weightUpdate[~self.sketchMask] = torch.sum(nonSketchedVs, dim=0)

        # for coords we are compressing, store the sum of the topk of vs
        sketchedVs = torch.stack(
                [topk(self.vs[workerId][self.sketchMask], k=self.opt.k)
                 for workerId in range(self.numWorkers)]
            )
        weightUpdate[self.sketchMask] = torch.sum(sketchedVs, dim=0)

        if self.opt.p2 == 1:
            # second round of communication
            HHs = weightUpdate.nonzero()
            weightUpdate[HHs] = sum([v[HHs] for v in self.vs])

        else:
            assert(self.opt.p2 == 0)
            # ALERT BAD CODE: this fragment below replaces what would
            # normally run in backward() for non-localtopk modes
            # but I can't do this there for local topk since after
            # returning the weightUpdate from this function we don't
            # know which coords of each v vector should be zeroed out
            for u, v, sv in zip(self.us, self.vs, sketchedVs):
                if self.opt.doAccumulateError:
                    # zero out just the coordinates that we sent
                    # don't zero out u for ~self.sketchMask since no nee
                    # to stop momentum for coords sent every iteration
                    sent = sv.nonzero()
                    u[self.sketchMask][sent] = 0
                    v[self.sketchMask][sent] = 0
                    v[~self.sketchMask] = 0

        return weightUpdate


    def _aggregateRandomVs(self):
        # we can only do this if we're compressing every coordinate
        # main reason is sampling without replacement from an arbitrary
        # subset of ~90M gradient coordinates takes a loong time
        # (~5 seconds in torch or numpy)
        assert(self.sketchMask.sum() == self.D)

        # choose a random set of k coordinates and send those
        # unfortunately currently no np.random.choice equivalent in torch
        # and unfortunately, np.random.choice is slooooooow for k<<d
        # fortunately, python's built-in random.sample is 50x faster....
        randomCoords = random.sample(range(self.D), self.opt.k)
        randomCoords = torch.tensor(randomCoords, device=self.device)

        weightUpdate = torch.zeros_like(self.vs[0])
        weightUpdate[randomCoords] = torch.sum(torch.stack(
                            [self.vs[workerId][randomCoords]
                             for workerId in range(self.numWorkers)]
                        ), dim=0)

        return weightUpdate

    #@profile
    def backward(self, doAggregate=True, flushVs=False):
        """Perform a backward pass, computing the gradient of the loss

        Args:
            doAggregate: whether or not to aggregate the workers'
                         gradients after computing them. Set to False
                         if, e.g., you plan to take a step on each worker
                         before sending the gradients back to the parameter
                         server.  (this is not really tested, sorry)
        """
        if flushVs:
            assert(doAggregate)
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
            if flushVs:
                weightUpdate = sum(self.vs)
            else:
                if self.doTrueTopk:
                    # for true top-k, just aggregate self.vs directly
                    weightUpdate = self._aggregateVs()
                    #print(torch.norm(weightUpdate))
                elif self.doLocalTopk:
                    weightUpdate = self._aggregateTopkVs()
                elif self.doRandomK:
                    weightUpdate = self._aggregateRandomVs()
                else:
                    # for sketched top-k, aggregate the sketches
                    weightUpdate = self._aggregateSketches()
                    #print(torch.norm(weightUpdate))

            #print(torch.stack(list(map(torch.std, self.vs))))
            # BAD CODE ALERT -- see note in _aggregateTopkVs
            localTopkException = self.doLocalTopk and (self.opt.p2 == 0)
            if not localTopkException:
                if self.opt.doAccumulateError:
                    # zero out coordinates on each worker that the
                    # parameter server updates
                    hhCoords = weightUpdate.nonzero()
                    #print("HH nonzero", hhCoords.size())
                    for workerId in range(self.numWorkers):
                        # the parameter server updated the heavy hitter
                        # coords and also the coords that don't get
                        # sketched
                        self.us[workerId][hhCoords] = 0
                        #self.us[workerId][~self.sketchMask] = 0
                        self.vs[workerId][hhCoords] = 0
                        self.vs[workerId][~self.sketchMask] = 0
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
        """Return the value of the loss"""
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
