from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .kubric import KubricDataset

dataset_dict = {
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
    'kubric': KubricDataset,
}
