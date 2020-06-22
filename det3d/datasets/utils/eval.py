import numpy as np
import numba

from det3d.ops.nms.nms_gpu import rotate_iou_gpu_eval
from det3d.ops.nms.nms_gpu import inter
from det3d.core import box_np_ops


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def prepare_data(gt_annos, dt_annos, current_class, difficulty=None, clean_data=None):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1
        )
        dt_datas = np.concatenate(
            [
                dt_annos[i]["bbox"],
                dt_annos[i]["alpha"][..., np.newaxis],
                dt_annos[i]["score"][..., np.newaxis],
            ],
            1,
        )
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        total_dc_num,
        total_num_valid_gt,
    )


def calculate_iou_partly(
    gt_annos, dt_annos, metric, num_parts=50, z_axis=1, z_center=1.0
):
    """fast iou algorithm. this function can be used independently to
    do result analysis.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
        z_axis: height axis. kitti camera use 1, lidar use 2.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    bev_axes = list(range(3))
    bev_axes.pop(z_axis)
    split_parts = [i for i in split_parts if i != 0]
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, bev_axes] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in gt_annos_part], 0
            )
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, bev_axes] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in dt_annos_part], 0
            )
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = box3d_overlap(
                gt_boxes, dt_boxes, z_axis=z_axis, z_center=z_center
            ).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][
                    gt_num_idx : gt_num_idx + gt_box_num,
                    dt_num_idx : dt_num_idx + dt_box_num,
                ]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


@numba.jit(nopython=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    dc_bboxes,
    metric,
    min_overlap,
    thresh=0,
    compute_fp=False,
    compute_aos=False,
):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (
                not compute_fp
                and (overlap > min_overlap)
                and dt_score > valid_detection
            ):
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (valid_detection == NO_DETECTION)
                and ignored_det[j] == 1
            ):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (
            ignored_gt[i] == 1 or ignored_det[det_idx] == 1
        ):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (
                assigned_detection[i]
                or ignored_det[i] == -1
                or ignored_det[i] == 1
                or ignored_threshold[i]
            ):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]:
                        continue
                    if ignored_det[j] == -1 or ignored_det[j] == 1:
                        continue
                    if ignored_threshold[j]:
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (
            query_boxes[k, 3] - query_boxes[k, 1]
        )
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]
                )
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                            + qbox_area
                            - iw * ih
                        )
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1, stable=False):
    if stable:
        riou = box_np_ops.riou_cc(boxes, qboxes)
    else:
        riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def box3d_overlap_kernel(boxes, qboxes, rinc, criterion=-1, z_axis=1, z_center=1.0):
    """
        z_axis: the z (height) axis.
        z_center: unified z (height) center of box.
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center),
                )
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center,
                )
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = area1 + area2 - inc
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def box3d_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = rotate_iou_gpu_eval(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    box3d_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc
