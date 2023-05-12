from models.blocks.utils import max_pool
import torch.nn as nn

class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return
    #
    # def forward(self, x, batch):
    #     return max_pool(x, batch['pools'][self.layer_ind + 1])
