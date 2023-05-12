from models.blocks.utils import closest_pool
import torch.nn as nn


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return
    #
    # def forward(self, x, batch):
    #     return closest_pool(x, batch['upsamples'][self.layer_ind - 1])
    #
    # def __repr__(self):
    #     return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
    #                                                               self.layer_ind - 1)
