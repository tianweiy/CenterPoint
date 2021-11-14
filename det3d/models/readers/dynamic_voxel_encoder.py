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

def voxelization_virtual(points, pc_range, voxel_size):    
    # current one is hard coded for nuScenes
    # TODO: fix those magic number 
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[keep, :]    

    real_points_mask = points[:, -2] == 1 
    painted_points_mask = points[:, -2] == 0 
    virtual_points_mask = points[:, -2] == -1 

    # remove zero padding for real points 
    real_points = points[real_points_mask][:, [0, 1, 2, 3, -1]]
    painted_point = points[painted_points_mask]  
    virtual_point = points[virtual_points_mask] 

    padded_points = torch.zeros(len(points), 22, device=points.device, dtype=points.dtype)

    # real points will occupy channels 0 to 4 and -1 
    padded_points[:len(real_points), :5] = real_points
    padded_points[:len(real_points), -1] = 1 

    # painted points will occupy channels 5 to 21 
    padded_points[len(real_points):len(real_points)+len(painted_point), 5:19] = painted_point[:, :-2]
    padded_points[len(real_points):len(real_points)+len(painted_point), 19] = painted_point[:, -1]
    padded_points[len(real_points):len(real_points)+len(painted_point), 20] = 1
    padded_points[len(real_points):len(real_points)+len(painted_point), 21] = 0

    #  virtual points will occupy channels 5 to 21 
    padded_points[len(real_points)+len(painted_point):, 5:19] = virtual_point[:, :-2]
    padded_points[len(real_points)+len(painted_point):, 19] = virtual_point[:, -1]
    padded_points[len(real_points)+len(painted_point):, 20] = 0
    padded_points[len(real_points)+len(painted_point):, 21] = 0

    points_xyz = torch.cat([real_points[:, :3], painted_point[:, :3], virtual_point[:, :3]], dim=0)

    coords = ((points_xyz[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) /  voxel_size[[2, 1, 0]]).to(torch.int64)
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)

    voxels = scatter_mean(padded_points, inverse_indices, dim=0)

    indicator = voxels[:, -1] 
    mix_mask = (indicator > 0) * (indicator < 1)
    # remove index 
    voxels = voxels[:, :-1] 

    voxels[mix_mask, :5] = voxels[mix_mask, :5] / indicator[mix_mask].unsqueeze(-1)
    voxels[mix_mask, 5:] = voxels[mix_mask, 5:] / (1-indicator[mix_mask].unsqueeze(-1))
    return voxels, unique_coords

@READERS.register_module
class DynamicVoxelEncoder(nn.Module):
    def __init__(
        self, pc_range, voxel_size, virtual=False
    ):
        super(DynamicVoxelEncoder, self).__init__()
        self.pc_range = torch.tensor(pc_range) 
        self.voxel_size = torch.tensor(voxel_size) 
        self.shape = torch.round((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size)
        self.shape_np = self.shape.numpy().astype(np.int32)
        self.virtual = virtual 

    @torch.no_grad()
    def forward(self, points):
        # points list[torch.Tensor]
        coors = []
        voxels = []  
        for res in points:
            if self.virtual:
                voxel, coor = voxelization_virtual(res, self.pc_range.to(res.device), self.voxel_size.to(res.device))
            else:
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

