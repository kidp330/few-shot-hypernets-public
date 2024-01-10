# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import pytorch_lightning as pl


def check_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Basic ResNet model


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(pl.LightningModule):
    def __init__(
        self,
        indim: int,
        outdim: int,
        scale_factor: int,
        class_wise_learnable_norm=True,
    ):
        super().__init__()
        self.L = nn.Linear(indim, outdim, bias=False)

        self.class_wise_learnable_norm = class_wise_learnable_norm
        if self.class_wise_learnable_norm:
            WeightNorm.apply(
                self.L, "weight", dim=0
            )  # split the weight update component to direction and norm

        self.scale_factor = scale_factor

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_L2 = torch.norm(x, dim=1)  # [v1, v2... vn] -> [||v1||, ||v2||, ... ||vn||]
        x_L2 = x_L2.unsqueeze(1).expand_as(
            x
        )  # -> [[||v1||], [||v2||], ...] -> repeat value along rows
        x_L2 += (eps := 1e-4)  # add eps~0 to avoid division by zero
        x_normalized = x.div(
            x_L2
        )  # each element of the feature vector is divided by the common norm
        return x_normalized

    # x is expected to have flat feature vectors
    def forward(self, x: torch.Tensor):
        x_normalized = self._normalize(x)
        if not self.class_wise_learnable_norm:
            self.L.weight.data = self._normalize(self.L.weigh.data)

        # matrix product by forward function, but when using WeightNorm,
        # this also multiplies the cosine distance by a class-wise learnable norm
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * cos_dist

        return scores


class Flatten(pl.LightningModule):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(
                x, self.weight.fast, self.bias.fast
            )  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super().forward(x)
        return out


class BLinear_fw(Linear_fw):  # used in BHMAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.weight.logvar = None
        self.weight.mu = None
        self.bias.logvar = None
        self.bias.mu = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            preds = []
            for w, b in zip(self.weight.fast, self.bias.fast):
                preds.append(F.linear(x, w, b))

            out = sum(preds) / len(preds)
        else:
            out = super(BLinear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(Conv2d_fw, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(
                    x, self.weight.fast, None, stride=self.stride, padding=self.padding
                )
            else:
                out = super().forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super().forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features, device: torch.device = check_device()):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1])
        running_var = torch.ones(x.data.size()[1])
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1,
            )
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1,
            )
        return out


# Simple Conv Block
class ConvBlock(pl.LightningModule):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# Simple ResNet Block
class SimpleBlock(pl.LightningModule):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False,
            )
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False,
            )
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, 2 if half_res else 1, bias=False
                )
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, 2 if half_res else 1, bias=False
                )
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = (
            x if self.shortcut_type == "identity" else self.BNshortcut(self.shortcut(x))
        )
        out = out + short_out
        out = self.relu2(out)
        return out


# Bottleneck block
class BottleneckBlock(pl.LightningModule):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
            )
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
            )
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [
            self.C1,
            self.BN1,
            self.C2,
            self.BN2,
            self.C3,
            self.BN3,
        ]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False
                )
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False
                )

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        short_out = x if self.shortcut_type == "identity" else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(pl.LightningModule):
    def __init__(self, depth, flatten=True, pool=False):
        super().__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if pool:
            trunk.append(nn.AdaptiveAvgPool2d((1, 1)))

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim: int = 64  # outdim if pool else 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetNopool(
    pl.LightningModule
):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super().__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(
                indim, outdim, pool=(i in [0, 1]), padding=0 if i in [0, 1] else 1
            )  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetS(
    pl.LightningModule
):  # For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten=True):
        super().__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        # trunk.append(nn.BatchNorm1d(64))    #TODO remove
        # trunk.append(nn.ReLU(inplace=True)) #TODO remove
        # trunk.append(nn.Linear(64, 64))     #TODO remove
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        # out = torch.tanh(out) #TODO remove
        return out


class ConvNetSNopool(
    pl.LightningModule
):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super().__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(
                indim, outdim, pool=(i in [0, 1]), padding=0 if i in [0, 1] else 1
            )  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 5, 5]

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


class ResNet(pl.LightningModule):
    maml = False  # Default

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super().__init__()
        assert len(list_of_num_layers) == 4, "Can have only four stages"
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


# Backbone for QMUL regression
class Conv3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 36, 3, stride=2, dilation=2)
        self.layer2 = nn.Conv2d(36, 36, 3, stride=2, dilation=2)
        self.layer3 = nn.Conv2d(36, 36, 3, stride=2, dilation=2)

    def return_clones(self):
        layer1_w = self.layer1.weight.data.clone().detach()
        layer2_w = self.layer2.weight.data.clone().detach()
        layer3_w = self.layer3.weight.data.clone().detach()
        return [layer1_w, layer2_w, layer3_w]

    def assign_clones(self, weights_list):
        self.layer1.weight.data.copy_(weights_list[0])
        self.layer2.weight.data.copy_(weights_list[1])
        self.layer3.weight.data.copy_(weights_list[2])

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = out.view(out.size(0), -1)
        return out


# just to test the kernel hypothesis
class BackboneKernel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_dim: int,
        flatten: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.flatten = flatten
        self.model = self.create_model()

    def create_model(self):
        assert self.num_layers >= 1, "Number of hidden layers must be at least 1"
        modules = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        if self.flatten:
            modules = [nn.Flatten()] + modules
        for _ in range(self.num_layers - 1):
            modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.hidden_dim, self.output_dim))

        model = nn.Sequential(*modules)
        return model

    def forward(self, x, **_params):
        r"""
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?
            :attr:`last_dim_is_batch` (tuple, optional):
                If this is true, it treats the last dimension of the data as another batch dimension.
                (Useful for additive structure over the dimensions). Default: False

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                * `diag`: `n` or `b x n`
                * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
        """
        out = self.model(x)

        return out


class ConvNet4WithKernel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        conv_out_size = 1600
        hn_kernel_layers_no = 4
        hn_kernel_hidden_dim = 64
        self.input_dim = conv_out_size
        self.output_dim = conv_out_size
        self.num_layers = hn_kernel_layers_no
        self.hidden_dim = hn_kernel_hidden_dim
        self.Conv4 = ConvNet(4)
        self.nn_kernel = BackboneKernel(
            self.input_dim, self.output_dim, self.num_layers, self.hidden_dim
        )
        self.final_feat_dim = self.output_dim

    def forward(self, x):
        x = self.Conv4(x)
        out = self.nn_kernel(x)
        return out


class ResNet10WithKernel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        conv_out_size = None
        hn_kernel_layers_no = None
        hn_kernel_hidden_dim = None
        self.input_dim = conv_out_size
        self.output_dim = conv_out_size
        self.num_layers = hn_kernel_layers_no
        self.hidden_dim = hn_kernel_hidden_dim
        self.Conv4 = ConvNet(4)
        self.nn_kernel = BackboneKernel(
            self.input_dim, self.output_dim, self.num_layers, self.hidden_dim
        )

    def forward(self, x):
        x = self.Conv4(x)
        x = torch.unsqueeze(torch.flatten(x), 0)
        out = self.nn_kernel(x)
        return out


def Conv4():
    return ConvNet(4)


def Conv4Pool():
    return ConvNet(4, pool=True)


def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


def Conv4S():
    return ConvNetS(4)


def Conv4SNP():
    return ConvNetSNopool(4)


def ResNet10(flatten=True):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten)


# def ResNet12(flatten=True):
# from learn2learn.vision.models import resnet12
#     class R12(pl.LightningModule):
#         def __init__(self):
#             super().__init__()
#             self.model = resnet12.ResNet12Backbone()
#             self.avgpool = nn.AvgPool2d(14)
#             self.flat = nn.Flatten()
#             self.final_feat_dim = 640  # 640

#         def forward(self, x):
#             x = self.model(x)
#             return x

#     return R12()


def ResNet18(flatten=True):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)


def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)


def ResNet50(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], [256, 512, 1024, 2048], flatten)


def ResNet101(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], [256, 512, 1024, 2048], flatten)


def Conv4WithKernel():
    return ConvNet4WithKernel()


def ResNetWithKernel():
    return ResNet10WithKernel()
