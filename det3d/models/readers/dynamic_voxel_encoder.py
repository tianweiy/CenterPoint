from det3d.core.utils.scatter import scatter_mean
from torch.nn import functional as F
from ..registry import READERS
from torch import nn
import numpy as np
import torch 

def voxelization(points, pc_range, voxel_size):    
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[keep, :]    
    coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) /  voxel_size[[2, 1, 0]]).to(torch.int64)
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)

    voxels = scatter_mean(points, inverse_indices, dim=0)
    return voxels, unique_coords

@READERS.register_module
class DynamicVoxelEncoder(nn.Module):
    def __init__(
        self, pc_range, voxel_size
    ):
        super(DynamicVoxelEncoder, self).__init__()
        self.pc_range = torch.tensor(pc_range) 
        self.voxel_size = torch.tensor(voxel_size) 
        self.shape = torch.round((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size)
        self.shape_np = self.shape.numpy().astype(np.int32)

    @torch.no_grad()
    def forward(self, points):
        # points list[torch.Tensor]
        coors = []
        voxels = []  
        for res in points:
            voxel, coor = voxelization(res, self.pc_range.to(res.device), self.voxel_size.to(res.device))
            voxels.append(voxel)
            coors.append(coor)

        coors_batch = [] 
        for i in range(len(voxels)):
            coor_pad = F.pad(coors[i], (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        voxels_batch = torch.cat(voxels, dim=0)
        return voxels_batch, coors_batch, self.shape_np

