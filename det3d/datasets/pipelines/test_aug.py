from det3d import torchie

from ..registry import PIPELINES
from .compose import Compose


@PIPELINES.register_module
class MultiScaleFlipAug(object):
    def __init__(self, transforms, img_scale, flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        assert torchie.is_list_of(self.img_scale, tuple)
        self.flip = flip

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                _results = results.copy()
                _results["scale"] = scale
                _results["flip"] = flip
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(transforms={}, img_scale={}, flip={})".format(
            self.transforms, self.img_scale, self.flip
        )
        return repr_str

@PIPELINES.register_module
class Flip(object):
    def __init__(self):
        pass

    def __call__(self, res, info):
        points = res["lidar"]["points"].copy()
        points[:, 1] = -points[:, 1]

        res["lidar"]['yflip_points'] = points

        return res, info 

@PIPELINES.register_module
class DoubleFlip(object):
    def __init__(self):
        pass

    def __call__(self, res, info):
        # y flip
        points = res["lidar"]["points"].copy()
        points[:, 1] = -points[:, 1]

        res["lidar"]['yflip_points'] = points

        # x flip
        points = res["lidar"]["points"].copy()
        points[:, 0] = -points[:, 0]

        res["lidar"]['xflip_points'] = points

        # x y flip
        points = res["lidar"]["points"].copy()
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]

        res["lidar"]["double_flip_points"] = points  

        return res, info 



