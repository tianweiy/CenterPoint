from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
