import torch.nn as nn
from models.blocks.batch_norm_block import BatchNormBlock


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Normalization
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        # multivariable linear
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return
    #
    # def forward(self, x):
    #     x = self.mlp(x)
    #     x = self.batch_norm(x)
    #     if not self.no_relu:
    #         x = self.leaky_relu(x)
    #     return x


class LastUnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        Initialize a standard last_unary block without BN, ReLU.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        """

        super(LastUnaryBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        return

    # def forward(self, x):
    #     x = self.mlp(x)
    #     return x
    #
    # def __repr__(self):
    #     return f'LastUnaryBlock(in_feat: {self.in_dim}, out_feat: {self.out_dim}'
