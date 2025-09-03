from typing import Union

from torch import nn
from torch.nn import functional as F

class FanInInitReLULayer(nn.Module):
    def __init__(
        self,
        inchan: int,
        outchan: int,
        norm_type: Union[str, None] = None,
        layer_type: str = "conv",
        use_activation: bool = True,
        **kwargs
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if norm_type:
            self.norm = dict(batch_norm=nn.BatchNorm2d, layer_norm=nn.LayerNorm)[norm_type](inchan)

        Layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = Layer(inchan, outchan, bias=self.norm is None, **kwargs)

        # # Init Weights (Fan-In)
        # self.layer.weight.data *= init_scale / self.layer.weight.norm(
        #     dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        # )
        # # Init Bias
        # if self.layer.bias is not None:
        #     self.layer.bias.data *= 0

        self.use_activation = use_activation

    def forward(self, x):
        """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

