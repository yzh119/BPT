import torch as th
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """
    Computer loss at one time step.
    """
    def __init__(self, size, smoothing=0.0):
        """Label Smoothing module
        args:
            size: vocab_size
            smoothing: smoothing ratio
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.size = size
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: (*, n_classes)
        # target: (*)
        assert x.size(1) == self.size
        with th.no_grad():
            tgt_dist = th.zeros_like(x, dtype=th.float)
            tgt_dist.fill_(self.smoothing / (self.size - 1))
            tgt_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        return self.criterion(x, tgt_dist)
