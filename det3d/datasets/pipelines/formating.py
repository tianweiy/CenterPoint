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

        if "anchors" in res["lidar"]["targets"]:
            anchors = res["lidar"]["targets"]["anchors"]
            data_bundle.update(dict(anchors=anchors))

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

        calib = res.get("calib", None)
        if calib:
            data_bundle["calib"] = calib

        if res["mode"] != "test":
            annos = res["lidar"]["annotations"]
            data_bundle.update(annos=annos, )

        if res["mode"] == "train":
            # ground_plane = res["lidar"].get("ground_plane", None)
            #if ground_plane:
            #    data_bundle["ground_plane"] = ground_plane

            if "reg_targets" in res["lidar"]["targets"]: # anchor based
                labels = res["lidar"]["targets"]["labels"]
                reg_targets = res["lidar"]["targets"]["reg_targets"]
                reg_weights = res["lidar"]["targets"]["reg_weights"]

                data_bundle.update(
                    dict(labels=labels, reg_targets=reg_targets, reg_weights=reg_weights)
                )
            else: # anchor free
                data_bundle.update(res["lidar"]["targets"])

        elif self.double_flip:
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
                annos=annos,  
            )
            if calib:
                yflip_data_bundle["calib"] = calib 

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
                annos=annos, 
            )
            if calib:
                xflip_data_bundle["calib"] = calib

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
                annos=annos, 
            )
            if calib:
                double_flip_data_bundle["calib"] = calib

            return [data_bundle, yflip_data_bundle, xflip_data_bundle, double_flip_data_bundle], info

        return data_bundle, info



@PIPELINES.register_module
class PointCloudCollect(object):
    def __init__(
            self,
            keys,
            meta_keys=(
                    "filename",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "img_norm_cfg",
            ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, info):

        results = info["res"]

        data = {}
        img_meta = {}

        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_meta"] = DC(img_meta, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(keys={}, meta_keys={})".format(
            self.keys, self.meta_keys
        )