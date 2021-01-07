import torch
from torch import nn

from ..registry import SECOND_STAGE
from det3d.core.utils.center_utils import (
    bilinear_interpolate_torch,
)

@SECOND_STAGE.register_module
class BEVFeatureExtractor(nn.Module): 
    def __init__(self, pc_start, 
            voxel_size, out_stride):
        super().__init__()
        self.pc_start = pc_start 
        self.voxel_size = voxel_size
        self.out_stride = out_stride

    def absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2

    def forward(self, example, batch_centers, num_point):
        batch_size = len(example['bev_feature'])
        ret_maps = [] 

        for batch_idx in range(batch_size):
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])
            
            # N x C 
            feature_map = bilinear_interpolate_torch(example['bev_feature'][batch_idx],
             xs, ys)

            if num_point > 1:
                section_size = len(feature_map) // num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(num_point)], dim=1)

            ret_maps.append(feature_map)

        return ret_maps 