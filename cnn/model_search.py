import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
  """
  Represents every possible operation at an edge.
  """
  def __init__(self, C, stride):
    """
    Constructor.
    :param C: int, number of channels.
    :param stride: int, stride of operation.
    """
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
  """
  Cells represent groups of operations connected in a particular manner.
  """
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    """
    Constructor.
    :param steps: int, number of nodes in the cell.
    :param multiplier: int, number of outputs to use on the final concatenation.
    :param C_prev_prev: int, channels of cell c_{k - 2}
    :param C_prev: int, channels of cell c_{k - 1}
    :param C: int, current number of channels.
    :param reduction: bool, is this a reduction cell?
    :param reduction_prev: bool, was the previous cell a reduction cell?
    """
    super(Cell, self).__init__()
    self.reduction = reduction

    # Choose a preprocessing step for state 0 taken from cell c_{k - 2}
    if reduction_prev:
      # If the previous cell was a reduction, increase the channels with two concatenated 2x2 convolutions instead of a 1x1.
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

    # Preprocessing step for cell c_{k - 1} is always a 1x1 conv.
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    # self._bns = nn.ModuleList()  # Not used anywhere in project.
    for i in range(self._steps):
      for j in range(2+i):
        # Only use stride 2 if this is a reduction cell and this operation is taking input from the input cells.
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
  """
  This network is trained to model the space of possible network architectures.
  """
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    """
    Constructor.
    :param C: int, number of input channels.
    :param num_classes: int, number of output classes.
    :param layers: int, total number of layers.
    :param criterion: nn._Loss, module that can calculate a loss function.
    :param steps: int, number of nodes inside a cell.
    :param multiplier: int, number of node outputs to depthwise concatenate at the output of the cell.
        Increases the number of output channels by multiplier * <current_channels>.
    :param stem_multiplier: int, increase the number of channels in the input by this factor.
    """
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    """
    Create a new Network and copy the architecture parameters into it.
    :return: Network
    """
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    """
    Forward pass method.
    Structure determined on forward call.
    :param input: torch.Tensor
    :return: torch.Tensor
    """
    # Get the initial states (input to the first cell).
    s0 = s1 = self.stem(input)

    # Iterate over all cells...
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)

      else:
        weights = F.softmax(self.alphas_normal, dim=-1)

      # Swap the states and get the new state from the current cell.
      s0, s1 = s1, cell(s0, s1, weights)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    """
    Initialize the parameters that determine the network architecture.
    """
    # Compute how many total connections the network has.
    # Old formula: k = sum(1 for i in range(self._steps) for _ in range(2+i))
    # Replaced with simpler closed form.
    k = (self._steps * (self._steps + 3)) // 2
    num_ops = len(PRIMITIVES)

    # Parameters arranged in a matrix.
    # k rows and num_ops columns.
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    """
    Get the genotype described the current architecture parameters.
    :return: Genotype | namedtuple, genotype that represents a cell, see genotype.py.
    """
    def _parse(weights):
      """
      Parse the weights that describe the architecture of the model.
      :param weights: torch.Tensor, rows are probability distributions (not actually necessary).
      :return: list, list of tuples with the following structure:
        [
            (<operation_string>, <index_of_input>),
            ...
        ]
      The ordering follows the numbering of nodes in the cell.
      In other words, the first 2 are for node 0, the next 3 are for node 1, etc...
      """
      gene = []
      n = 2
      start = 0
      # For connection in the cell described by the alpha matrix...
      for i in range(self._steps):
        end = start + n

        # Get the rows that correspond to the alpha vectors that govern the current node in the cell.
        W = weights[start:end].copy()

        # Determine the two connections that are going to be used.
        # This weird sort lambda gets the indices of the two rows that have the largest values.
        # This determines the states that will be inputs to this node.
        edges = sorted(
            range(i + 2),
            key=lambda x: \
                                                       # This if statement disallows ever picking a "none" connection.
                                                       # This functions as not connecting the nodes.
                -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none'))
        )[:2]

        # For each of the edges we just determined, append their corresponding operation and index to the gene.
        for j in edges:
          k_best = None
          # Get the index of the maximum value in the aplpha vector.
          for k in range(len(W[j])):
            # None check allows for explicitly not having a connection between two nodes.
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          # Create the primitive and append it to the genome.
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    # Do a softmax along the each row to obtain a probability distribution for each operation in the architecture.
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    # Get the range of nodes to concatenate at the end.
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

