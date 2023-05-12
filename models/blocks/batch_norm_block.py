import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Normalization
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim

        if self.use_bn:
            # self.batch_norm = nn.BatchNorm1d(in_dim, momentum = bn_momentum)
            self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)

        else:
            # 가중치 0으로 초기화
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    #
    # def forward(self, x):
    #     if self.use_bn:
    #         # unsqueeze 함수 특정 차원에 1인 차원을 추가함
    #         # unsqueeze(2)는 (5,123,4) -> (5,123,2,4)
    #         x = x.unsqueeze(2)
    #         x = x.transpose(0, 2)
    #         x = self.batch_norm(x)
    #         x = x.transpose(0, 2)
    #         # x.squeeze는 1인 차원을 지우는 용도
    #         return x.squeeze()
    #
    # def __repr__(self):
    #     return f'BatchNormBlock(in_feat : {self.in_dim}, momenturm: {self.bn_momentum}, ' \
    #            f'only_bias:{str(not self.use_bn)}'
