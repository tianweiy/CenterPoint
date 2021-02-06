
import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import yaml
import time

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

from det3d import __version__, torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

import cupy as cp
from collections import deque
from copy import deepcopy
from functools import reduce


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0, 0, 1], radians=yaw)

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()

    car_indices = get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices = get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(
        2, 0.4, label_preds_, scores_)
    bus_indices = get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices = get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices = get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices = get_annotations_indices(
        6, 0.15, label_preds_, scores_)
    bicycle_indices = get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices = get_annotations_indices(
        8, 0.12, label_preds_, scores_)
    traffic_cone_indices = get_annotations_indices(
        9, 0.1, label_preds_, scores_)

    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices +
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None

        self.lidar_deque = deque(maxlen=5)
        self.current_frame = {
            "lidar_stamp": None,
            "lidar_seq": None,
            "points": None,
            "odom_seq": None,
            "odom_stamp": None,
            "translation": None,
            "rotation": None
        }
        self.pc_list = deque(maxlen=5)
        self.inputs = None

    def initialize(self):
        self.read_config()

    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[1],
        )
        # nuscenes dataset
        lidar2imu_t = np.array([0.985793, 0.0, 1.84019])
        lidar2imu_r = Quaternion([0.706749235, -0.01530099378, 0.0173974518, -0.7070846])

        ## UDI dataset
        # lidar2imu_t = np.array([1.50, 0., 1.42])
        # lidar2imu_r = Quaternion([1., 0., 0., 0.])
        self.lidar2imu = transform_matrix(lidar2imu_t, lidar2imu_r, inverse=True)
        self.imu2lidar = transform_matrix(lidar2imu_t, lidar2imu_r, inverse=False)

    def run(self):

        # print(f"input points shape: {points.shape}")
        # num_features = 5
        # self.points = points.reshape([-1, num_features])

        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)),
                        mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(
            num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(
            num_voxels, dtype=torch.int32, device=self.device)
        # grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)

        # t = time.time()
        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]  # simulate a batch of one example 
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]

        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        outputs = remove_low_score_nu(outputs, 0.45)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        return scores, boxes_lidar, types

    def get_lidar_data(self, input_points: dict):
        print("get one frame lidar data.")
        self.current_frame["lidar_stamp"] = input_points['stamp']
        self.current_frame["lidar_seq"] = input_points['seq']
        self.current_frame["points"] = input_points['points'].T   
        self.lidar_deque.append(deepcopy(self.current_frame))
        if len(self.lidar_deque) == 5:

            ref_from_car = self.imu2lidar
            car_from_global = transform_matrix(self.lidar_deque[-1]['translation'], self.lidar_deque[-1]['rotation'], inverse=True)

            ref_from_car_gpu = cp.asarray(ref_from_car)
            car_from_global_gpu = cp.asarray(car_from_global)

            for i in range(len(self.lidar_deque) - 1):
                last_pc = self.lidar_deque[i]['points']
                last_pc_gpu = cp.asarray(last_pc)

                global_from_car = transform_matrix(self.lidar_deque[i]['translation'], self.lidar_deque[i]['rotation'], inverse=False)
                car_from_current = self.lidar2imu
                global_from_car_gpu = cp.asarray(global_from_car)
                car_from_current_gpu = cp.asarray(car_from_current)

                transform = reduce(
                    cp.dot,
                    [ref_from_car_gpu, car_from_global_gpu, global_from_car_gpu, car_from_current_gpu],
                )
                # tmp_1 = cp.dot(global_from_car_gpu, car_from_current_gpu)
                # tmp_2 = cp.dot(car_from_global_gpu, tmp_1)
                # transform = cp.dot(ref_from_car_gpu, tmp_2)

                last_pc_gpu = cp.vstack((last_pc_gpu[:3, :], cp.ones(last_pc_gpu.shape[1])))
                last_pc_gpu = cp.dot(transform, last_pc_gpu)

                self.pc_list.append(last_pc_gpu[:3, :])

            current_pc = self.lidar_deque[-1]['points']
            current_pc_gpu = cp.asarray(current_pc)
            self.pc_list.append(current_pc_gpu[:3,:])

            all_pc = np.zeros((5, 0), dtype=float)
            for i in range(len(self.pc_list)):
                tmp_pc = cp.vstack((self.pc_list[i], cp.zeros((2, self.pc_list[i].shape[1]))))
                tmp_pc = cp.asnumpy(tmp_pc)
                ref_timestamp = self.lidar_deque[-1]['lidar_stamp'].to_sec()
                timestamp = self.lidar_deque[i]['lidar_stamp'].to_sec()
                tmp_pc[3, ...] = self.lidar_deque[i]['points'][3, ...]
                tmp_pc[4, ...] = ref_timestamp - timestamp
                all_pc = np.hstack((all_pc, tmp_pc))
            
            all_pc = all_pc.T
            print(f" concate pointcloud shape: {all_pc.shape}")

            self.points = all_pc
            sync_cloud = xyz_array_to_pointcloud2(all_pc[:, :3], stamp=self.lidar_deque[-1]["lidar_stamp"], frame_id="lidar_top")
            pub_sync_cloud.publish(sync_cloud)
            return True

    def get_odom_data(self, input_odom):

        self.current_frame["odom_stamp"] = input_odom.header.stamp
        self.current_frame["odom_seq"] = input_odom.header.seq
        x_t = input_odom.pose.pose.position.x
        y_t = input_odom.pose.pose.position.y
        z_t = input_odom.pose.pose.position.z
        self.current_frame["translation"] = np.array([x_t, y_t, z_t])
        x_r = input_odom.pose.pose.orientation.x
        y_r = input_odom.pose.pose.orientation.y
        z_r = input_odom.pose.pose.orientation.z
        w_r = input_odom.pose.pose.orientation.w
        self.current_frame["rotation"] = Quaternion([w_r, x_r, y_r, z_r])


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(
            cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    points[..., 3] = cloud_array['intensity']
    return points


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    # msg.data = points_sum.astype(np.float32).tobytes()
    return msg


def rslidar_callback(msg):
    # t_t = time.time()
    arr_bbox = BoundingBoxArray()
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
 
    print("  ")
    seq = msg.header.seq
    stamp = msg.header.stamp
    input_points = {
        'stamp': stamp,
        'seq': seq,
        'points': np_p
    }
    if(proc_1.get_lidar_data(input_points)):
        scores, dt_box_lidar, types = proc_1.run()

        if scores.size != 0:
            for i in range(scores.size):
                bbox = BoundingBox()
                bbox.header.frame_id = msg.header.frame_id
                bbox.header.stamp = rospy.Time.now()
                q = yaw2quaternion(float(dt_box_lidar[i][8]))
                bbox.pose.orientation.x = q[1]
                bbox.pose.orientation.y = q[2]
                bbox.pose.orientation.z = q[3]
                bbox.pose.orientation.w = q[0]
                bbox.pose.position.x = float(dt_box_lidar[i][0])
                bbox.pose.position.y = float(dt_box_lidar[i][1])
                bbox.pose.position.z = float(dt_box_lidar[i][2])
                bbox.dimensions.x = float(dt_box_lidar[i][4])
                bbox.dimensions.y = float(dt_box_lidar[i][3])
                bbox.dimensions.z = float(dt_box_lidar[i][5])
                bbox.value = scores[i]
                bbox.label = int(types[i])
                arr_bbox.boxes.append(bbox)
        # print("total callback time: ", time.time() - t_t)
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp
        if len(arr_bbox.boxes) is not 0:
            pub_arr_bbox.publish(arr_bbox)
            arr_bbox.boxes = []
        else:
            arr_bbox.boxes = []
            pub_arr_bbox.publish(arr_bbox)


def odom_callback(msg):
    '''
    get odom data
    '''
    proc_1.get_odom_data(msg)


if __name__ == "__main__":

    global proc
    # CenterPoint
    config_path = 'configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms_demo.py'
    model_path = 'models/last.pth'

    proc_1 = Processor_ROS(config_path, model_path)

    proc_1.initialize()

    rospy.init_node('centerpoint_ros_node')
    sub_lidar_topic = ["/velodyne_points",
                       "/top/rslidar_points",
                       "/points_raw",
                       "/aligned/point_cloud",
                       "/merged_cloud",
                       "/lidar_top",
                       "/roi_pclouds"]
    sub_lidar = rospy.Subscriber(
        sub_lidar_topic[5], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)

    sub_odom_topic = ["/golfcar/odom",
                      "/aligned/odometry",
                      "/odom"]

    sub_odom = rospy.Subscriber(
        sub_odom_topic[2], Odometry, odom_callback, queue_size=10, buff_size=2**10, tcp_nodelay=True)

    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)
    pub_sync_cloud = rospy.Publisher("sync_5sweeps_cloud", PointCloud2, queue_size=1)

    print("[+] CenterPoint ros_node has started!")
    rospy.spin()
