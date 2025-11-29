from torch import nn
from typing import Union, Optional
from torch.nn import functional as F
from .impala_cnn import ImpalaCNN
from .relu_layer import FanInInitReLULayer
from .norm import DynamicNormFix, ImgNorm
from .mlp_blocks import ResidualMLPBlock


class DualEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        raise NotImplementedError


class VectorObsProcess(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pre_dnorm: bool = True
    ):
        super().__init__()
        self.batch_norm = None
        if pre_dnorm:
            self.batch_norm = DynamicNormFix(input_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
        self.obs_encoder = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(inplace=True), nn.Linear(output_dim, output_dim))
        self.mlp = ResidualMLPBlock(output_dim)
    
    def forward(self, obs, test_freeze_dnorm: bool = False):
        if self.batch_norm:
            obs = self.batch_norm.forward(obs, freeze=test_freeze_dnorm)
        return self.mlp(self.obs_encoder(obs))


class ImgObsProcess(nn.Module):
    def __init__(
        self,
        imgshape_chw: tuple = (3, 128, 128),
        impala_chans: tuple = (16, 32, 32),
        cnn_outsize: int = 256,
        output_size: int = 512,
        pre_norm: bool = False
    ):
        super().__init__()
        assert len(imgshape_chw) == 3, str(imgshape_chw)
        self.norm = None
        if pre_norm: self.norm = ImgNorm(scale_img=True, norm_img=False)
        self.cnn = ImpalaCNN(
            inshape_chw=imgshape_chw,
            chans=impala_chans,
            outsize=cnn_outsize,
            nblock=2
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
        )

    def forward(self, img):
        if self.norm: img = self.norm(img)
        return self.linear(self.cnn(img))
