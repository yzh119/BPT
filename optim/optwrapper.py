from functools import partial
import math


class NoamOpt(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        model_size: hidden size
        factor: coefficient
        warmup: warm up steps(step ** (-0.5) == step * warmup ** (-1.5) holds when warmup equals step)
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5))
            )

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def adjust_lr(self, *args):
        pass

class CosineOpt(object):
    def __init__(self, optimizer, T_max=83000, warmup=4000):
        self.optimizer = optimizer
        self._step = 0
        self.T_max = T_max
        self.warmup = warmup
        self.init_lr = {}
        for p in self.optimizer.param_groups:
            self.init_lr[id(p)] = p['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            for p in self.optimizer.param_groups:
                p['lr'] = self.init_lr[id(p)] * (self._step / self.warmup) 
        else:
            for p in self.optimizer.param_groups:
                p['lr'] = self.init_lr[id(p)] * (1 + math.cos(math.pi * (self._step - 1) / self.T_max)) / 2 # TODO
        self.optimizer.step()

    def adjust_lr(self, *args, **kwargs):
        pass        

class ReduceLROnPlateauOpt(object):
    def __init__(self, optimizer, order='min', rate=0.5, warmup=1600):
        self.optimizer = optimizer
        self.order = order
        self._step = 0
        self.warmup = warmup
        self.best = 1e9 if order == 'min' else -1e9
        self.rate = rate
        self.init_lr = {}
        for p in self.optimizer.param_groups:
            self.init_lr[id(p)] = p['lr']

    @classmethod
    def get_instance(cls, order='min'):
        def func(optimizer, rate=0.5):
            return ReduceLROnPlateauOpt(optimizer, order=order, rate=rate)
        return func

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update(self, val):
        if val < self.best and self.order == 'min':
            self.best = val
            return True

        if val > self.best and self.order == 'max':
            self.best = val
            return True

        return False

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            for p in self.optimizer.param_groups:
                p['lr'] = self.init_lr[id(p)] * (self._step / self.warmup) 
        self.optimizer.step()

    def adjust_lr(self, val):
        if not self.update(val):
            for p in self.optimizer.param_groups:
                p['lr'] = p['lr'] * self.rate


class ExponentialDecay(object):
    def __init__(self, optimizer, rate=0.99, warmup=4000):
        self.optimizer = optimizer
        self.rate = rate
        self._step = 0
        self.warmup = warmup
        self.init_lr = {}
        for p in self.optimizer.param_groups:
            self.init_lr[id(p)] = p['lr']


    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            for p in self.optimizer.param_groups:
                p['lr'] = self.init_lr[id(p)] * (self._step / self.warmup) 
        self.optimizer.step()

    def adjust_lr(self, *args):
        for p in self.optimizer.param_groups:
            p['lr'] = p['lr'] * self.rate


class Steady(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
   
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def adjust_lr(self, *args):
        pass


def get_wrapper(name):
    if name == 'noam':
        return NoamOpt
    if name == 'cosine':
        return CosineOpt 
    elif name == 'exp_decay':
        return ExponentialDecay
    elif name == 'steady':
        return Steady
    elif name == 'reduce_lr_on_plateau_min':
        return ReduceLROnPlateauOpt.get_instance(order='min')
    elif name == 'reduce_lr_on_plateau_max':
        return ReduceLROnPlateauOpt.get_instance(order='max')

