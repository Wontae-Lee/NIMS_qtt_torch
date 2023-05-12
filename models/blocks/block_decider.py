from models.blocks.simple_block import SimpleBlock
from models.blocks.unary_block import *
from models.blocks.resnet_bottle_neck_block import ResnetBottleneckBlock
from models.blocks.max_pool_block import MaxPoolBlock
from models.blocks.global_average_block import GlobalAverageBlock
from models.blocks.nearest_upsample_block import NearestUpsampleBlock


def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):
    if block_name == "unary":
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)
    if block_name == 'last_unary':
        return LastUnaryBlock(in_dim, config.final_feats_dim + 2)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)
