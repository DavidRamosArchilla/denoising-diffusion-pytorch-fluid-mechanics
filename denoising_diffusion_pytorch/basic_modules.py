import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

# borrowed from LightningDiT https://github.com/hustvl/LightningDiT/blob/main/models/swiglu_ffn.py
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        bias=True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    # @torch.compile
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

REGISTERED_ACT_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "relu": (nn.ReLU, {"inplace": True}),
    "relu6": (nn.ReLU6, {"inplace": True}),
    "hswish": (nn.Hardswish, {"inplace": True}),
    "hsigmoid": (nn.Hardsigmoid, {"inplace": True}),
    "swish": (nn.SiLU, {"inplace": True}),
    "silu": (nn.SiLU, {"inplace": True}),
    "tanh": (nn.Tanh, {}),
    "sigmoid": (nn.Sigmoid, {}),
    "gelu": (nn.GELU, {"approximate": "tanh"}),
    "mish": (nn.Mish, {"inplace": True}),
    "identity": (nn.Identity, {}),
}


def build_act(name: str or None, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls, default_args = copy.deepcopy(REGISTERED_ACT_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return act_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError(f"do not support: {name}")


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2

    
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,
        use_bias=False,
        dropout=0.0,
        conv_type="1d",
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        if conv_type == "1d":
            self.conv = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
            )
        elif conv_type == "2d":
            self.conv = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=padding,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        elif conv_type == "3d":
            self.conv = nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                stride=(stride, stride, stride),
                padding=padding,
                dilation=(dilation, dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        else:
            self.conv = None
        # no me importa porque siempre va a ser None
        # self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

# taken from https://github.dev/NVlabs/Sana
class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,
        use_bias=(False, False, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        # use_bias = val2tuple(use_bias, 3)
        # norm = val2tuple(norm, 3)
        # act = val2tuple(act, 3)

        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x, HW=None):
        """esto funciona solo para 1d"""
        # B, N, C = x.shape
        # if HW is None:
        #     H = W = int(N**0.5)
        # elif len(HW) == 2:
        #     H, W = HW
        #     x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # elif len(HW) == 3:
        #     T, H, W = HW
        #     x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x = x.permute(0, 2, 1) # B, C, N
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        # if len(HW) == 3:
        #     x = x.reshape(B * T, C, H * W).permute(0, 2, 1)
        #     x = x.reshape(B, N, C)
        # else:
        #     x = x.reshape(B, C, N).permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        return x