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
    def __init__(self, model, sketchBiases=False, sketchParamsLargerThan=0):
        self.model = model
        # sketch everything larger than sketchParamsLargerThan
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketchParamsLargerThan

        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketchBiases

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

        # used for debugging
        self._doSlowSketching = False

        # self.modelDevice is not tested... not sure what happens if
        # the model is on the CPU
        if opt.param_groups[0]["params"][0].is_cuda:
            self.modelDevice = "cuda"
        else:
            self.modelDevice = "cpu"
        self.device = self.modelDevice
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
                                         device=self.device,
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
            self.losses.append(loss[start:end].sum())
        return self

    def _backwardWorker(self, workerId):
        """Do a backward pass for one worker

        Args:
            workerId: which worker to do the backward pass for (between
                      0 and self.numWorkers - 1)
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

        # [MOMENTUM_TYPE] this is the way Karimireddy+2019 ("error feedback
        # fixed signSGD...") combine error feedback with the LR.
        # In pytorch, on the other hand, torch.nn.SGD multiples v,
        # not g by the LR. To use the pytorch method, uncomment these
        # two lines and see the other MOMENTUM_TYPE comment.
        #lrVec = self._getLRVec()
        #gradVec *= lrVec

        if self.opt.doAccumulateError:
            self.us[workerId].mul_(self.opt.momentum).add_(gradVec)
            self.vs[workerId] += self.us[workerId]
        else:
            self.vs[workerId] += gradVec

    # the helper functions below deal only with the compressed coordinates
    def _aggAndZeroTrueTopk(self):
        weightUpdate = torch.zeros_like(self.vs[0])
        vs = [v[self.sketchMask] for v in self.vs]
        w = topk(torch.sum(torch.stack(vs), dim=0), k=self.opt.k)
        weightUpdate[self.sketchMask] = w
        for u, v in zip(self.us, self.vs):
            # zeroing u won't do anything if accumulateError is False
            u[weightUpdate.nonzero()] = 0
            v[weightUpdate.nonzero()] = 0
        return weightUpdate

    def _aggAndZeroLocalTopk(self):
        weightUpdate = torch.zeros_like(self.vs[0])
        if self.opt.p2 is None or self.opt.p2 == 0:
            for u, v in zip(self.us, self.vs):
                ltk = topk(v, k=self.opt.k)
                # no second round of communication
                # weightUpdate is just the sum of localTopks,
                weightUpdate[self.sketchMask] += ltk
                # and each worker zeros out only what it sent

                # want to do v[sketchMask][ltk.nonzero()] = 0
                # but this doesn't work since v[sketchMask] makes
                # a copy, and then only the copy gets zeroed
                sent = self.sketchMask.clone().float()
                sent[self.sketchMask] *= ltk
                # can do nonzero() since sent is size self.D
                u[sent.nonzero()] = 0 # momentum stopping
                v[sent.nonzero()] = 0 # reset error accumulation
        else:
            localTopks = [topk(v[self.sketchMask], k=self.opt.k)
                          for v in self.vs]
            # do a second round of communication

            # doesn't make sense to request more than Wk coordinates
            # in the second round of communication, since the PS would
            # just have to choose the additional coords at random
            assert(self.opt.p2 <= self.numWorkers)
            # do a second round of communication to get true
            # values of the top k*p2 coords among those that were
            # sent from any worker
            hhs = torch.sum(torch.stack(localTopks), dim=0)
            del localTopks
            hhs = topk(hhs, self.opt.p2*self.opt.k).nonzero()
            #for workerId in range(self.numWorkers):
            #    print("WORKER {} LTK: ".format(workerId), localTopks[workerId])
            w = torch.sum(torch.stack([v[self.sketchMask][hhs]
                                       for v in self.vs]), dim=0)
            # roundabout way to do weightUpdate[sketchMask][hhs] = w
            sent = torch.zeros_like(weightUpdate[self.sketchMask])
            sent[hhs] = w
            weightUpdate[self.sketchMask] = sent
            # and then zero out all weightUpdate.nonzero(),
            # since w now contains the values from each worker
            # for every weightUpdate.nonzero() coord
            for u, v in zip(self.us, self.vs):
                u[weightUpdate.nonzero()] = 0
                v[weightUpdate.nonzero()] = 0
        return weightUpdate

    def _aggAndZeroRandomK(self):
        # instead of sampling k elements, sample k + # uncompressed
        # do this because ideally we'd send all the uncompressed ones
        # plus k of the compressed ones. But we can't sample from only
        # the uncompressed ones (see comment below), so we sample from
        # all of them, which could theoretically lead us to send as few
        # as k - # uncompressed if we happen to choose all the uncompressed
        # coords. So instead, be conservative and sample k + # uncompressed
        numCoords = self.opt.k + (~self.sketchMask).nonzero().numel()

        # choose a random set of numCoords coordinates and send those
        # unfortunately, sampling without replacement (using torch,
        # np, or python's built-in random) takes several seconds for
        # typical inputs. So instead, sample slightly more than k coords,
        # and then choose the unique ones. How many is "slightly" more?
        # compute the expected number of unique elements E_k when drawing
        # k' from [1..D]. Then solve for k', setting E_k=numCoords.
        # this is a bit ridiculous, but it's 1000x faster than
        # the alternatives...
        # Note: we won't get exactly numCoords unique draws, but it's
        # pretty close, and we're already fudging it with the
        # non-compressed coords (see below)
        # Note2: we should do torch.unique here, but we have to
        # do torch.unique below anyway, so no point doing it twice
        nSamples = np.log(1-numCoords/self.D) / np.log((self.D-1)/self.D)
        nSamples = int(nSamples)
        randomCoords = torch.randint(int(self.D),
                                     size=(nSamples,),
                                     device=self.device)

        # ideally, we would've sampled from self.sketchMask.nonzero(). But
        # sampling without replacement from an arbitrary
        # subset of ~90M gradient coordinates takes a loong time
        # (~5 seconds in torch or numpy)
        # so instead, we sampled from [0..D-1], and now we just force that
        # all non-compressed coordinates are included
        uncompressedCoords = (~self.sketchMask).nonzero().view(-1)
        toSend = torch.cat((randomCoords, uncompressedCoords))
        # we might have sampled an uncompressed coord, so take unique elems
        # this is fast as long as it's on the GPU (~150x faster than CPU)
        toSend = torch.unique(toSend)

        w = torch.sum(torch.stack([v[toSend] for v in self.vs]), dim=0)
        weightUpdate = torch.zeros_like(self.vs[0])
        weightUpdate[toSend] = w
        for u, v in zip(self.us, self.vs):
            # toSend \in (0, D), so this is a reasonable indexing
            u[toSend] = 0
            v[toSend] = 0

        return weightUpdate

    def _sketchHelper(self, vs):
        # sketch vs into self.workerSketches
        # this method does it the proper way, by sketching each
        # v into the corresponding sketch
        for workerId, v in enumerate(vs):
            # zero the sketch from the previous round
            self.workerSketches[workerId].zero()
            if self.opt.p1 > 0:
                # truncate and then sketch
                tk = topk(v, self.opt.p1 * self.opt.k)
                self.workerSketches[workerId] += tk
            else:
                # sketch without truncating
                self.workerSketches[workerId] += v

    def _sketchHelperShortcut(self, vs):
        # sketch the sum of vs into self.workerSketches[0]
        # this produces the same output as sketching each of the
        # numWorkers v vectors into the corresponding sketch, and then
        # summing the sketches. But you only have to sketch a single
        # vector (doing this is only possible in the simulated
        # distributed environment)
        for workerId in range(self.numWorkers):
            self.workerSketches[workerId].zero()
        if self.opt.p1 > 0:
            # truncate each vector
            summed = sum([topk(v, self.opt.p1 * self.opt.k) for v in vs])
        else:
            summed = sum(vs)
        self.workerSketches[0] += summed

    def _aggAndZeroSketched(self):
        """Aggregate the sketches of each worker

        If p2 > 0, do a second round of communication between the
        parameter server and the workers in order to get a better
        estimate of the topk (both which elements are in the topk and
        the values of those elements)
        """
        if self.sketchMask.sum() < self.vs[0].numel():
            vs = [v[self.sketchMask] for v in self.vs]
        else:
            vs = self.vs

        # caller can turn on using the slow version for testing
        # purposes by setting self._doSlowSketching = True
        if self._doSlowSketching is not None and self._doSlowSketching:
            self._sketchHelper(vs)
        else:
            self._sketchHelperShortcut(vs)

        # now gather workerSketches, and do a 2nd round of communication
        # if p2 > 0
        if self.opt.p2 > 0:
            candidateTopk = np.sum(self.workerSketches).unSketch(
                                k=self.opt.p2*self.opt.k)
            # get coords that were populated by the unSketch
            # (i.e. the heavy hitters)
            candidateHHCoords = candidateTopk.nonzero()
            # get exact values for candidateHHCoords
            for v in vs:
                candidateTopk[candidateHHCoords] += v[candidateHHCoords]
            del vs
            w = topk(candidateTopk, k=self.opt.k)
            del candidateTopk
            weightUpdate = torch.zeros_like(self.vs[0])
            weightUpdate[self.sketchMask] = w
        else:
            # if p2 == 0, then there's no second round of
            # communication: we just use the values for the gradient
            # that we got from the sketch
            assert(self.opt.p2 == 0)
            w = np.sum(self.workerSketches).unSketch(k=self.opt.k)
            weightUpdate = torch.zeros_like(self.vs[0])
            weightUpdate[self.sketchMask] = w

        # zero out the coords of u, v that are being updated
        for u, v in zip(self.us, self.vs):
            u[weightUpdate.nonzero()] = 0
            v[weightUpdate.nonzero()] = 0

        if False:
            # just for debugging
            trueWeightUpdate = topk(sum(self.vs), k=self.opt.k)
            overlap = torch.sum((weightUpdate != 0) * (trueWeightUpdate != 0)).item()
            print("OVERLAP:", overlap, "out of ", self.opt.k)
            print("(nonzero WU):", weightUpdate.nonzero().size())

        return weightUpdate


    def _aggregateAndZeroUVs(self):

        # first, deal with just the compressed coordinates
        # (delegated to helper functions)
        if self.doTrueTopk:
            weightUpdate = self._aggAndZeroTrueTopk()
        elif self.doLocalTopk:
            weightUpdate = self._aggAndZeroLocalTopk()
        elif self.doRandomK:
            weightUpdate = self._aggAndZeroRandomK()
        else:
            weightUpdate = self._aggAndZeroSketched()

        # now deal with the non-compressed coordinates
        # Note: this is a no-op if doRandomK=True, since we dealt
        # with the uncompressed coords already in self._aggAndZeroRandomK()
        vs = [v[~self.sketchMask] for v in self.vs]
        weightUpdate[~self.sketchMask] = torch.sum(torch.stack(vs), dim=0)
        for v in self.vs:
            # only zero out v, not u -- we don't want to stop momentum
            # for coords that are being updated every iteration
            v[~self.sketchMask] = 0

        # reset the error accumulation vector every time if
        # error accumulation is turned off
        if not self.opt.doAccumulateError:
            for v in self.vs:
                v.zero_()

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

        # give backwardWorker a clean slate to backprop into
        self._setGradVec(torch.zeros_like(initialGradVec))

        # backprop on each worker, updating self.us and self.vs
        for workerId in range(self.numWorkers):
            self._backwardWorker(workerId)

        # doAggregate is True when we're ready to make a step
        if doAggregate:
            if flushVs:
                weightUpdate = sum(self.vs)
                for u, v in zip(self.us, self.vs):
                    u.zero_()
                    v.zero_()
            else:
                weightUpdate = self._aggregateAndZeroUVs()
            # add back the initial gradient vector
            weightUpdate.add_(initialGradVec)

            # [MOMENTUM_TYPE] This is how torch.optim.SGD does momentum
            # (different from Karimireddy+2019).
            # To use Karimireddy+2019's method instead, swap commented
            # lines below and see other MOMENTUM_TYPE comment
            self._setGradVec(weightUpdate * self._getLRVec())
            #self._setGradVec(weightUpdate)
        else:
            # if we're not aggregating, then put back the initialGradVec
            # (since self._backwardWorker may have modified it)
            self._setGradVec(initialGradVec)

        # return the number of parameters that get updated, so whoever
        # called us knows how much communication happened
        return weightUpdate.nonzero().numel()


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
