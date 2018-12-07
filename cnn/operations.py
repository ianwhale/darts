import math
import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride), # Seems to never actually be used...
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_1x3_3x1' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),
    nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'conv_3x3': lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, affine=affine),
  'lbc_3x3': lambda C, stride, affine: LBCLayer(C, C, stride, affine=affine, sparsity=0.1),
  'pert_3x3': lambda C, stride, affine: PerturbationLayer(C, C, affine=affine)  # Doesn't use stride.
}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

#
# Local binary convolution code taken from the original authors:
#   https://github.com/human-analysis/pytorchnet/blob/master/models/lbcresnet.py
#

def lbcconv3x3(in_planes, out_planes, stride=1, sparsity=0.1):
    """
    Local Binary Convolution layer.
    See original paper for more info:
      http://openaccess.thecvf.com/content_cvpr_2017/papers/Juefei-Xu_Local_Binary_Convolutional_CVPR_2017_paper.pdf
    :param in_planes: int, number of input channels.
    :param out_planes: int, number of output channels.
    :param stride: int, stride.
    :param sparsity: float, the sparsity of the binary convolution [0, 1].
    :return: nn.Conv2d, with special binary weights.
    """
    conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                       stride=stride, padding=1, bias=False)
    conv2d.weight.requires_grad = False
    conv2d.weight.fill_(0.0)
    num = conv2d.weight.numel()
    index = torch.Tensor(math.floor(sparsity * num)).random_(num).int()
    conv2d.weight.resize_(in_planes * out_planes * 3 * 3)
    for i in range(index.numel()):
        conv2d.weight[index[i]] = torch.bernoulli(torch.Tensor([0.5])) * 2 - 1

    conv2d.weight.resize_(out_planes, in_planes, 3, 3)
    return conv2d


class LBCLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride, affine=True, sparsity=0.1):
        """
        Constructor.
        :param in_planes: int, number of input channels.
        :param out_planes: int, number of output channels.
        :param stride: int, stride.
        :param affine: bool, use affine in batchnorm?
        :param sparsity: float, sparsity value for binary convolution.
        """
        super(LBCLayer, self).__init__()
        self.layers = nn.Sequential(
            lbcconv3x3(in_planes, out_planes, stride, sparsity),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes, affine=affine)  # Small change from original adds batch norm too LBCLayer.
        )

    def forward(self, x):
        z = self.layers(x)
        return z

#
# Perturbative neural network code taken from the original authors:
#   https://github.com/juefeix/pnn.pytorch.update/blob/master/models.py#L443
#   Note: renamed NoiseLayer to PerturbationLayer for clarity's sake.

class PerturbationLayer(nn.Module):
    """
    Perturbative neural network layer.
    See original paper for more:
      http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0200.pdf
    """
    def __init__(self, in_planes, out_planes, affine=True, level=0.1):
        """
        Constructor.
        :param in_planes: int, input channels.
        :param out_planes: int, output channels.
        :param affine: bool, should the batchnorm use affine?
        :param level: float, noise multiplier, see forward method.
        """
        super(PerturbationLayer, self).__init__()

        # TODO: Check if this gives an error with multiple GPUs.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_planes, affine=affine),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()   #fill with uniform noise
            self.noise = (2 * self.noise - 1) * self.level
        y = torch.add(x, self.noise)
        return self.layers(y)   #input, perturb, relu, batchnorm, conv1x1
