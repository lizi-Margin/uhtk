import math, torch
from typing import Dict, List, Optional

from torch import nn
from torch.nn import functional as F

from .utils import sequential, transpose, flatten_last3dims
from uhtk.siri.utils.iterable_eq import iterable_prod as intprod
from .relu_layer import FanInInitReLULayer

from uhtk.print_pack import *

class CnnBasicBlock(nn.Module):
    def __init__(
        self,
        inchan: int
    ):
        super().__init__()
        self.inchan = inchan
        self.conv0 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            layer_type='conv',
            kernel_size=3,
            padding=1,
        )
        self.conv1 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            layer_type='conv',
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = x + self.conv1(self.conv0(x))
        return x


class CnnDownStack(nn.Module):
    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        pool: bool = True,
    ):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            layer_type='conv',
            kernel_size=3,
            padding=1,
        )
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                )
                for _ in range(nblock)
            ]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = sequential(self.blocks, x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)

class ImpalaCNN(nn.Module):
    def __init__(
        self,
        inshape: List[int],
        chans: List[int],
        outsize: int,
        nblock: int,
    ):
        super().__init__()
        h, w, c = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                pool=True
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        
        cnnout_flatten_size = intprod(curshape)
        if (outsize < cnnout_flatten_size) or (outsize > 2*cnnout_flatten_size):
            print黄(lprint_(self, f"Warning: mapping output size \
                            from {cnnout_flatten_size} to {outsize}."))

        self.dense = FanInInitReLULayer(
            cnnout_flatten_size,
            outsize,
            layer_type="linear",
        )
        self.outsize = outsize

    def forward(self, x: torch.Tensor):
        to_batch = x.shape[:-3]
        x = x.reshape(intprod(to_batch), *x.shape[-3:])
        x = transpose(x, "bhwc", "bchw")
        x = sequential(self.stacks, x)
        x = x.reshape(*to_batch, *x.shape[1:])
        x = flatten_last3dims(x)
        x = self.dense(x)
        return x
