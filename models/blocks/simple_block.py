from models.blocks.kernel_points_conv import KPConv
from models.blocks.batch_norm_block import BatchNormBlock
import torch.nn as nn


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # Get kp_extent from current radius
        current_extent = radius * config.kp_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(config.kernel_size,
                             config.nfeatures_pts,
                             in_dim,
                             out_dim // 2,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             kp_influence=config.kp_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)

        # Other opperations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return
    #
    # def forward(self, x, batch):
    #
    #     if 'strided' in self.block_name:
    #         q_pts = batch['points'][self.layer_ind + 1]
    #         s_pts = batch['points'][self.layer_ind]
    #         neighb_inds = batch['pools'][self.layer_ind]
    #     else:
    #         q_pts = batch['points'][self.layer_ind]
    #         s_pts = batch['points'][self.layer_ind]
    #         neighb_inds = batch['neighbors'][self.layer_ind]
    #
    #     x = self.KPConv(q_pts, s_pts, neighb_inds, x)
    #     return self.leaky_relu(self.batch_norm(x))
