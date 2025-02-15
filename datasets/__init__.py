from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .kubric import KubricDataset, KubricFlowDataset

dataset_dict = {
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
    'kubric': KubricDataset,
    'kubric_flow': KubricFlowDataset,
}
