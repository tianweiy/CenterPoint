from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        double_flip = kwargs.get('double_flip', False)
        self.double_flip = double_flip 

    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]

        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"]
        )

        if res["mode"] == "train":
            data_bundle.update(res["lidar"]["targets"])
        elif res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

            if self.double_flip:
                # y axis 
                yflip_points = res["lidar"]["yflip_points"]
                yflip_voxels = res["lidar"]["yflip_voxels"] 
                yflip_data_bundle = dict(
                    metadata=meta,
                    points=yflip_points,
                    voxels=yflip_voxels["voxels"],
                    shape=yflip_voxels["shape"],
                    num_points=yflip_voxels["num_points"],
                    num_voxels=yflip_voxels["num_voxels"],
                    coordinates=yflip_voxels["coordinates"],
                )

                # x axis 
                xflip_points = res["lidar"]["xflip_points"]
                xflip_voxels = res["lidar"]["xflip_voxels"] 
                xflip_data_bundle = dict(
                    metadata=meta,
                    points=xflip_points,
                    voxels=xflip_voxels["voxels"],
                    shape=xflip_voxels["shape"],
                    num_points=xflip_voxels["num_points"],
                    num_voxels=xflip_voxels["num_voxels"],
                    coordinates=xflip_voxels["coordinates"],
                )
                # double axis flip 
                double_flip_points = res["lidar"]["double_flip_points"]
                double_flip_voxels = res["lidar"]["double_flip_voxels"] 
                double_flip_data_bundle = dict(
                    metadata=meta,
                    points=double_flip_points,
                    voxels=double_flip_voxels["voxels"],
                    shape=double_flip_voxels["shape"],
                    num_points=double_flip_voxels["num_points"],
                    num_voxels=double_flip_voxels["num_voxels"],
                    coordinates=double_flip_voxels["coordinates"],
                )

                return [data_bundle, yflip_data_bundle, xflip_data_bundle, double_flip_data_bundle], info


        return data_bundle, info



