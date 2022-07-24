import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.use_depth = kwargs.get('use_depth', False)
        self.define_transforms()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.rays)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            idxs = np.random.choice(len(self.rays), self.batch_size)
            sample = {
                'rays': self.rays[idxs, :6],
                'rgb': self.rays[idxs, 6:9],
                'idxs': idxs,
            }
            if self.use_depth:
                sample['disp'] = self.rays[idxs, 9]
        else:
            sample = {
                'rays': self.rays[idx][:, :6],
                'rgb': self.rays[idx][:, 6:9],
                'idx': idx,
            }

        return sample
