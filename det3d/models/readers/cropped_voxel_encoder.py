from collections import defaultdict

import numpy as np
import torch
from det3d.core.bbox.box_np_ops import points_in_rbbox, riou_cc, rotation_3d_in_axis
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.torchie.trainer.utils import get_dist_info


def crop2assign(
    example, predictions, cfg=None, device=torch.device("cpu"), training=True
):
    rank, world_size = get_dist_info()
    # STEP 2: crop enlarged point clouds; organise as new batch'
    # 2.1 Prepare batch targets, filter invalid target
    batch_targets = []
    # 2.1 Prepare batch data
    batch_data = defaultdict(list)

    if training:
        gt_boxes3d = defaultdict(list)

        for idx, sample in enumerate(example["annos"]):
            for branch in sample["gt_boxes"]:
                mask = np.ma.masked_equal(branch, 0).mask
                if np.any(mask):
                    gt_boxes3d[idx].extend(branch[np.where(mask.sum(axis=1) == 0)])
                else:
                    gt_boxes3d[idx].extend(branch)

    # Batch size
    cnt = 0
    for idx, dt_boxes3d in enumerate(predictions):
        sample_points = (
            example["points"][example["points"][:, 0] == idx][:, 1:].cpu().numpy()
        )
        if sample_points[:, -1].min() < 0:
            sample_points[:, -1] += 0.5

        if training:
            gp = example["ground_plane"][idx]

            sample_gt_boxes3d = np.array(gt_boxes3d[idx])
            if not sample_gt_boxes3d.shape[0] == 0:
                gt_bevs = sample_gt_boxes3d[:, [0, 1, 3, 4, -1]]
            else:
                gt_bevs = np.zeros((0, 5))

        sample_dt_boxed3d = dt_boxes3d["box3d_lidar"].cpu().numpy()

        if training:
            dt_bevs = sample_dt_boxed3d[:, [0, 1, 3, 4, -1]]

            gp_height = cfg.anchor.center - cfg.anchor.height / 2

            # Find max match gt
            ious = riou_cc(gt_bevs, dt_bevs)
            dt_max_matched_gt = ious.argmax(axis=0)
            max_ious = ious.max(axis=0)

            selected = np.where(max_ious >= 0.7)

            max_ious = max_ious[selected]
            dt_max_matched_gt = dt_max_matched_gt[selected]
            # remove fp
            sample_dt_boxed3d = sample_dt_boxed3d[selected]

        # enlarge box
        sample_dt_boxed3d[:, [3, 4]] += cfg.roi_context

        indices = points_in_rbbox(sample_points, sample_dt_boxed3d)
        num_points_in_gt = indices.sum(0)
        # remove empty dt boxes
        selected_by_points = np.where(num_points_in_gt > 0)

        boxes = sample_dt_boxed3d[selected_by_points]
        indices = indices.transpose()[selected_by_points].transpose()

        if training:
            dt_max_matched_gt = dt_max_matched_gt[selected_by_points]
            cnt += len(boxes)

        # voxel_generators to form fixed_size batch input
        fixed_size = cfg.dense_shape  # w, l, h

        for i, box in enumerate(boxes):

            if training:
                batch_data["ground_plane"].append(gp)

                # 1. generate regression targets
                matched_gt = sample_gt_boxes3d[dt_max_matched_gt[i]]

                height_a = cfg.anchor.height
                z_center_a = cfg.anchor.center
                # z_g = matched_gt[2] + matched_gt[5]/2. - (-0.14) # z top
                z_g = matched_gt[2] - z_center_a
                h_g = matched_gt[5] - height_a
                g_g = (matched_gt[2] - matched_gt[5] / 2) - (float(gp))
                dt_target = np.array([idx, i, z_g, h_g, g_g])

                batch_targets.append(dt_target)

            if not training:
                batch_data["stage_one_output_boxes"].append(box)

            # 2. prepare data

            box_points = sample_points[indices[:, i]].copy()
            # img = kitti_vis(box_points, np.array(box).reshape(1, -1))
            # img.tofile(open("./bev.bin", "wb"))

            # move to center
            box_center = box[:2]
            box_points[:, :2] -= box_center
            # rotate to canonical
            box_yaw = box[-1:]
            box_points_canonical = rotation_3d_in_axis(
                box_points[:, :3][np.newaxis, ...], -box_yaw, axis=2
            )[0]
            box_points_canonical = np.hstack((box_points_canonical, box_points[:, -1:]))

            if box_points_canonical.shape[0] > 0:
                point_cloud_range = [
                    -box[3] / 2,
                    -box[4] / 2,
                    -3.5,
                    box[3] / 2,
                    box[4] / 2,
                    1.5,
                ]
            else:
                import pdb

                pdb.set_trace()

            voxel_size = [
                (point_cloud_range[3] - point_cloud_range[0]) / fixed_size[0],
                (point_cloud_range[4] - point_cloud_range[1]) / fixed_size[1],
                (point_cloud_range[5] - point_cloud_range[2]) / fixed_size[2],
            ]
            for vs in voxel_size:
                if not vs > 0:
                    import pdb

                    pdb.set_trace()

            vg = VoxelGenerator(
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                max_num_points=20,
                max_voxels=2000,
            )

            voxels, coordinates, num_points = vg.generate(
                box_points_canonical, max_voxels=2000
            )
            batch_data["voxels"].append(voxels)
            batch_data["coordinates"].append(coordinates)
            batch_data["num_points"].append(num_points)

    if training:
        batch_targets = torch.tensor(
            np.array(batch_targets), dtype=torch.float32, device=device
        )
        batch_data["targets"] = batch_targets

        batch_data["ground_plane"] = torch.tensor(
            batch_data["ground_plane"], dtype=torch.float32, device=device
        )

    for k, v in batch_data.items():
        if k in ["voxels", "num_points"]:
            batch_data[k] = np.concatenate(v, axis=0)
        elif k in ["stage_one_output_boxes"]:
            batch_data[k] = np.array(v)
        elif k in ["coordinates"]:
            coors = []
            for i, coor in enumerate(v):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            batch_data[k] = np.concatenate(coors, axis=0)

    for k, v in batch_data.items():
        if k in ["coordinates", "num_points"]:
            batch_data[k] = torch.tensor(v, dtype=torch.int32, device=device)
        elif k in ["voxels", "stage_one_output_boxes"]:
            batch_data[k] = torch.tensor(v, dtype=torch.float32, device=device)

    if training:
        if (
            not cnt
            == (batch_data["coordinates"][:, 0].max() + 1)
            == batch_data["targets"].shape[0]
        ):
            import pdb

            pdb.set_trace()

    batch_data["shape"] = fixed_size

    return batch_data
