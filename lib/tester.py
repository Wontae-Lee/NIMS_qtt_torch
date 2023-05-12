import os
import torch
from tqdm import tqdm
from lib.trainer import Trainer


class IndoorTester(Trainer):
    """
    3DMatch tester
    """

    def __init__(self, config):
        Trainer.__init__(self, config)

    def test(self):
        print('Start to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)

        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):  # loop through this epoch
                inputs = c_loader_iter.next()
                ##################################
                # load inputs to device.
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)
                ###############################################
                # forward pass
                feats, scores_overlap, scores_saliency = self.model(inputs)  # [N1, C1], [N2, C2]
                pcd = inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']

                src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                data = dict()
                data['pcd'] = pcd.cpu()
                data['feats'] = feats.detach().cpu()
                data['overlaps'] = scores_overlap.detach().cpu()
                data['saliency'] = scores_saliency.detach().cpu()
                data['len_src'] = len_src
                data['rot'] = c_rot.cpu()
                data['trans'] = c_trans.cpu()

                torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')


def get_trainer(config):
    if config.dataset == 'indoor':
        return IndoorTester(config)
    elif config.dataset == 'kitti':
        return
    elif config.dataset == 'modelnet':
        return
    else:
        raise NotImplementedError
