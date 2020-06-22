import numpy as np
import torch


def example_to_device(
    example, dtype=torch.float32, device=None, non_blocking=True
) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "reg_targets", "reg_weights", "labels", "anchors_mask"]:
            res = []
            for kk, vv in v.items():
                vv = [vvv.unsqueeze_(0) for vvv in vv]
                res.append(torch.cat(vv, dim=0).cuda(device, non_blocking=non_blocking))
            example_torch[k] = res
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
        ]:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = v.cuda(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.cuda(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])
