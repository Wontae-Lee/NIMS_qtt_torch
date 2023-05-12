import torch
import os


class Config:

    def __init__(self):
        # misc:
        self.exp_dir = "indoor"
        self.mode = "train"
        self.gpu_mode = True
        self.verbose = True
        self.verbose_freq = 1000
        self.snapshot_freq = 1
        self.pretrain = ''

        # model:
        self.num_layers = 4
        self.nfeatures_pts = 3
        self.first_feats_dim = 128
        self.final_feats_dim = 32
        self.first_subsampling_dl = 0.025
        self.in_feats_dim = 1
        self.conv_radius = 2.5
        self.deform_radius = 5.0
        self.kernel_size = 15
        self.kp_extent = 2.0
        self.kp_influence = "linear"
        self.aggregation_mode = "sum"
        self.fixed_kernel_points = "center"
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02
        self.deformable = False
        self.modulated = False
        self.add_cross_score = True
        self.condition_feature = True

        # overlap_attention_module:
        self.gnn_feats_dim = 256
        self.dgcnn_k = 10
        self.num_head = 4
        self.nets = ['self', 'cross', 'self']

        # loss:
        self.pos_margin = 0.1
        self.neg_margin = 1.4
        self.log_scale = 24
        self.pos_radius = 0.0375
        self.safe_radius = 0.1
        self.overlap_radius = 0.0375
        self.matchability_radius = 0.05
        self.w_circle_loss = 1.0
        self.w_overlap_loss = 1.0
        self.w_saliency_loss = 0.0
        self.max_points = 256

        # optimiser:
        self.optimizer = "SGD"
        self.max_epoch = 40
        self.lr = 0.005
        self.weight_decay = 0.000001
        self.momentum = 0.98
        self.scheduler = "ExpLR"
        self.scheduler_gamma = 0.95
        self.scheduler_freq = 1
        self.iter_size = 1

        # dataset:
        self.dataset = "indoor"
        self.benchmark = "3DMatch"
        self.root = "data/indoor"
        self.batch_size = 1
        self.num_workers = 6
        self.augment_noise = 0.005
        self.train_info = "./configs/indoor/train_info.pkl"
        self.val_info = "./configs/indoor/val_info.pkl"

        # demo:
        self.src_pcd = "assets/cloud_bin_21.pth"
        self.tgt_pcd = "assets/cloud_bin_34.pth"
        self.n_points = 1000

        # additional argument
        self.snapshot_dir = f'snapshot/{self.exp_dir}'
        self.tboard_dir = f'snapshot/{self.exp_dir}/tensorboard'
        self.save_dir = f'snapshot/{self.exp_dir}/checkpoints'

        # make directories
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tboard_dir, exist_ok=True)

        # choose mode
        if self.gpu_mode:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # undefined parameters
        self.architecture = None
        self.model = None
        self.desc_loss = None
