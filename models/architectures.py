import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_  # 가중치 초기화
from models.blocks import *


def regular_scores(score):
    # score가 nan수이거나 inf수 이면 0으로 만든다.
    score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
    return score


class KPFCNN(nn.Module):

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        '''
        Parameters
        '''

        layer = 0

        # sampling rate
        # ?
        r = config.first_subsampling_dl * config.conv_radius

        # input dimension -> 3d coordinates -> 3
        in_dim = config.in_feats_dim

        # output dimension 입력층의 출력 차원
        out_dim = config.first_feats_dim

        # torch.tensor(-5.0): -5.0라는 값만 가지고 있는 텐서
        # torch.nn.Parameter: 레이어가 아니라 파라미터
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

        # final features dimension
        self.final_feats_dim = config.final_feats_dim

        # condition feature
        # ?
        self.condition = config.condition_feature

        # cross overlap excution True or False
        self.add_cross_overlap = config.add_cross_score

        '''
        List Encoder blocks
        '''
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # check equivariance
            if ('equivarient' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsamle', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            # encoder에 신경망 추가
            self.encoder_blocks.append()
