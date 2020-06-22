import argparse
import binascii
import json
import linecache
import os
import struct
import sys
from random import choice

import numpy as np
import vtk
from vtk_visualizer import pointobject, visualizercontrol

try:
    import cPickle
except:
    import pickle as cPickle


color = {
    "car": (0, 255, 0),  # green
    "pedestrian": (255, 128, 128),  # pink
    "bicycle": (0, 0, 255),  # bule
    "unknown": (0, 255, 255),  # yellow
}

# fake camera
# Tr = np.asarray([[0,-1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])


class VtkPointCloud:
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)

        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(-255, 255)
        mapper.SetScalarVisibility(1)

        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = np.random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName("DepthArray")
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars("DepthArray")


# def project_velo2camera(vel_data, Tr):
#     # vel_data_c: col 0: back -> front
#     #             col 1: down -> up
#     #             col 2: left -> right
#     homo_vel_data = np.hstack((vel_data[:,:3],np.ones((vel_data.shape[0],1), dtype='float32')))
#     vel_data_c = np.dot(homo_vel_data, Tr.T)
#     vel_data_c /= vel_data_c[:, -1].reshape((-1,1))
#     vel_data_c = np.hstack((vel_data_c[:, :3], vel_data[:, -1].reshape((-1,1))))
#     return vel_data_c
#
#
# def read_bin(bin_file):
#     points = np.fromfile(bin_file,dtype=np.float32).reshape([-1,5])[:, :4]
#     points_c = project_velo2camera(points, Tr)
#     return points_c[:,:3]
#
# def read_pickle(bin_file):
#     with open(bin_file, 'rb') as f:
#         result = cPickle.load(f)
#     lidar = np.asarray(result[b'points'])
#     lidar = lidar[~np.isnan(lidar[:,0])]
#     lidar_c = project_velo2camera(lidar, Tr)
#     return lidar_c[:,:3]
#
# def read_label(label_path):
#     label = [line for line in open(label_path, 'r').readlines()]
#     boxes_center = []
#     for line in label:
#         ret = line.strip().split()
#         # if ret[0] == 'Car':
#         h, w, l, x, y, z, r = [float(i) for i in ret[8:15]]
#         boxes_center.append([x, y, z, h, w, l, r])
#     return center_to_corner_box3d(boxes_center)
#
# def center_to_corner_box3d(boxes_center):
#     # (N, 7) -> (N, 8, 3)
#     N = len(boxes_center)
#     ret = np.zeros((N, 8, 3), dtype=np.float32)
#
#     for i in range(N):
#         box = boxes_center[i]
#         translation = box[0:3]
#         size = box[3:6]
#         rotation = [0, 0, box[-1]]
#
#         h, w, l = size[0], size[1], size[2]
#         trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
#             [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], \
#             [0,0,0,0,-h,-h,-h,-h], \
#             [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]])
#
#         yaw = rotation[2]
#         rotMat = np.array([
#             [np.cos(yaw), 0.0, np.sin(yaw)],
#             [0.0, 1.0, 0.0],
#             [-np.sin(yaw), 0.0, np.cos(yaw)]])
#         cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
#         box3d = cornerPosInVelo.transpose()
#         ret[i] = box3d
#     return ret


def rbbox3d_to_corners(rbboxes, origin=[0.5, 0.5, 0.5], axis=2):
    return center_to_corner_box3d(
        rbboxes[..., :3], rbboxes[..., 3:6], rbboxes[..., 6], origin, axis=axis
    )


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def generator_color(length, aim):
    color_length = np.array(aim).shape[-1]
    color = np.full((length, color_length), np.asarray(aim))
    return color


def reset_coord(tot_obj, cloud, co):
    final = []
    for i in range(tot_obj):
        for j in range(cloud):
            pass


def mkVtkIdList(it):
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


pts = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def draw_box(x):
    cube = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()

    for i in range(8):
        points.InsertPoint(i, x[i])
    for i in range(6):
        polys.InsertNextCell(mkVtkIdList(pts[i]))
    for i in range(8):
        scalars.InsertTuple1(i, i)

    cube.SetPoints(points)
    del points
    cube.SetPolys(polys)
    del polys
    cube.GetPointData().SetScalars(scalars)
    del scalars

    cubeMapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        cubeMapper.SetInput(cube)
    else:
        cubeMapper.SetInputData(cube)
    cubeMapper.SetScalarRange(0, 7)
    # cubeMapper.SetScalarVisibility(2)
    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetOpacity(0.4)
    return cubeActor


def draw_grid():
    xArray = vtk.vtkDoubleArray()
    yArray = vtk.vtkDoubleArray()
    zArray = vtk.vtkDoubleArray()

    for x in range(-60, 61):
        xArray.InsertNextValue(x)
    # for y in range(-10, 10):
    #     yArray.InsertNextValue(y)
    for y in range(0, 1):
        yArray.InsertNextValue(y)
    for z in range(-80, 80):
        zArray.InsertNextValue(z)

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(121, 1, 161)
    grid.SetXCoordinates(xArray)
    grid.SetYCoordinates(yArray)
    grid.SetZCoordinates(zArray)

    # print(grid.GetPoint(0))

    gridMapper = vtk.vtkDataSetMapper()
    gridMapper.SetInputData(grid)

    gridActor = vtk.vtkActor()
    gridActor.SetMapper(gridMapper)
    gridActor.GetProperty().SetColor(0.75, 0.75, 0)
    gridActor.GetProperty().SetOpacity(0.1)
    # import pdb; pdb.set_trace()
    return gridActor


def render(args):

    # file_list = os.listdir(os.path.join(args.dataset_path, 'prediction'))
    # file_name = choice(file_list)[:-4]
    # oripoints = read_pickle(os.path.join(args.dataset_path, 'lidar', file_name+'.pkl'))
    # boxes_corner = read_label(os.path.join(args.dataset_path, 'prediction', file_name+'.txt'))

    # for KITTI
    # oripoints = read_bin(args.bin_path)
    # for Hobot
    # oripoints = read_pickle(args.bin_path)
    # boxes_corner = read_label(args.txt_path)

    oripoints = np.fromfile("./data_labels/10_points.bin", dtype=np.float32).reshape(
        [-1, 5]
    )[:, 1:]
    bboxes = np.fromfile("./data_labels/10_boxes_f64.bin", dtype=np.float64).reshape(
        [-1, 7]
    )

    boxes_corner = rbbox3d_to_corners(bboxes)

    ren = vtk.vtkRenderer()
    for box in boxes_corner:
        cubeActor = draw_box(box)
        ren.AddActor(cubeActor)

    gridActor = draw_grid()
    ren.AddActor(gridActor)

    colors1 = generator_color(oripoints.shape[0], [128, 128, 128])
    obj = pointobject.VTKObject()
    obj.CreateFromArray(np.array(oripoints))
    obj.AddColors(colors1)

    ren.AddActor(obj.GetActor())
    axes = vtk.vtkAxesActor()
    grid = vtk.vtkImageGridSource()
    axes.SetTotalLength(80, 10, 80)

    ren.AddActor(axes)
    ren.SetBackground(0.1, 0.2, 0.3)

    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1600, 1600)
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()

    # #start from this line
    # vtkControl = visualizercontrol.VTKVisualizerControl()
    # vtkControl.AddPointCloudActor(oripoints)
    # nID = vtkControl.GetLastActorID()
    # vtkControl.SetActorColor(nID, (255, 255, 255))

    # # for i in range(len(points)):
    # #     vtkControl.AddPointCloudActor(np.asarray(points[i]))
    # #     nID = vtkControl.GetLastActorID()
    # #     vtkControl.SetActorColor(nID, names[i])

    # vtkControl.ResetCamera()

    # renWin = vtk.vtkRenderWindow()
    # renWin.AddRenderer(ren)
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)
    # renWin.Render()
    # iren.Start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Python",
        description="Aim at testing cnn_seg results through VTK rendering",
    )
    parser.add_argument(
        "--bin-path", type=str, required=True, help="path for lidar file"
    )
    parser.add_argument("--txt-path", type=str, required=True, help="path for txt file")
    # parser.add_argument("--dataset-path", type=str, required=True, help="path for dataset")
    args = parser.parse_args()

    render(args)
