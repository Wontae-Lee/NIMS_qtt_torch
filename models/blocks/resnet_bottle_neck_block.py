from models.blocks.kernel_points_conv import *
from models.blocks.unary_block import UnaryBlock
from models.blocks.batch_norm_block import BatchNormBlock
class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get kp_extent from current radius
        current_extent = radius * config.kp_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(config.kernel_size,
                             config.nfeatures_pts,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             kp_influence=config.kp_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return
    #
    # def forward(self, features, batch):
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
    #     # First downscaling mlp
    #     x = self.unary1(features)
    #
    #     # Convolution
    #     x = self.KPConv(q_pts, s_pts, neighb_inds, x)
    #     x = self.leaky_relu(self.batch_norm_conv(x))
    #
    #     # Second upscaling mlp
    #     x = self.unary2(x)
    #
    #     # Shortcut
    #     if 'strided' in self.block_name:
    #         shortcut = max_pool(features, neighb_inds)
    #     else:
    #         shortcut = features
    #     shortcut = self.unary_shortcut(shortcut)
    #
    #     return self.leaky_relu(x + shortcut)
