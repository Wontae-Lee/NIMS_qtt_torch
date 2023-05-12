import torch.nn as nn
from models.blocks.utils import global_average


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()
        return
    #
    # def forward(self, x, batch):
    #     return global_average(x, batch['stack_lengths'][-1])
