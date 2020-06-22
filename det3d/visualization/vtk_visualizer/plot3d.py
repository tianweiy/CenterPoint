# -*- coding: utf-8 -*-
"""
Helper functions for easy visualization of point clouds

.. Author: Øystein Skotheim, SINTEF ICT <oystein.skotheim@sintef.no>
   Date:   Thu Sep 12 15:50:40 2013
"""

import random

import vtk_visualizer.visualizercontrol as vis

g_vtk_control = None
g_hold = False


__colors = {
    "r": (1, 0, 0),
    "g": (0, 1, 0),
    "b": (0, 0, 1),
    "c": (0, 1, 1),
    "m": (1, 0, 1),
    "y": (1, 1, 0),
    "w": (1, 1, 1),
    #'k': (0, 0, 0),  # do we need black?
}


def _next_color(color):
    c = list(__colors.keys())
    i = c.index(color)
    i += 1
    if i >= len(c):
        i = 0
    return c[i]


def _char2color(c):
    "Convert one-letter code to an RGB color"
    global __colors
    return __colors[c]


def get_vtk_control(block):
    "Get the vtk control currently used by the plot3d functions"
    global g_vtk_control

    if block or g_vtk_control is None:
        g_vtk_control = vis.VTKVisualizerControl()

    return g_vtk_control


def toggle_hold(flag=True):
    "Toggle whether new objects will replace previous objects in the visualizer"
    global g_hold
    g_hold = flag


def is_hold_enabled():
    "Returns whether hold is enabled"
    global g_hold
    return g_hold


def plotxyz(pts_l, color="g", hold=False, block=False):
    """Plot a supplied point cloud (NumPy array of Nxd values where d>=3)

    An optional color may be given as a single character (rgbcmykw)
    An optional hold flag can be enabled to keep previously visualized data

    The supplied array may have an additional column with scalars, which
    will be used to color the points (0=black, 1=white)
    """
    if not isinstance(pts_l, list):
        pts_l = [(pts_l)]
    vtkControl = get_vtk_control(block)
    if not (hold or is_hold_enabled()):
        vtkControl.RemoveAllActors()

    for pts in pts_l:
        vtkControl.AddPointCloudActor(pts)
        if pts.shape[1] <= 3:
            nID = vtkControl.GetLastActorID()
            vtkControl.SetActorColor(nID, _char2color(color))
            vtkControl.Render()
            color = _next_color(color)

    if block:
        vtkControl.exec_()


def plotxyzrgb(pts, hold=False, block=False):
    "Plot a supplied point cloud w/ color (NumPy array of Nx6 values)"
    vtkControl = get_vtk_control(block)
    if not (hold or is_hold_enabled()):
        vtkControl.RemoveAllActors()
    vtkControl.AddColoredPointCloudActor(pts)
    vtkControl.Render()
    if block:
        vtkControl.exec_()


def plothh(pts, scale=5.0, hold=False, block=False):
    "Plot hedge hog (points w/ normals) from given NumPy array of Nx6 values"
    vtkControl = get_vtk_control(block)
    if not (hold or is_hold_enabled()):
        vtkControl.RemoveAllActors()
    vtkControl.AddHedgeHogActor(pts, scale)
    vtkControl.Render()
    if block:
        vtkControl.exec_()


if __name__ == "__main__":

    import numpy as np

    h, w = 256, 256

    [y, x] = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 10 * np.cos(0.1 * np.sqrt((x - w / 2.0) ** 2 + (y - h / 2.0) ** 2))

    pc1 = np.zeros((h, w, 3))
    pc1[:, :, 0] = x
    pc1[:, :, 1] = y
    pc1[:, :, 2] = z

    pc2 = pc1.copy()
    pc2[:, :, 2] += 10.0
    rgb = 255 * np.random.rand(h, w, 3)

    pc2 = np.dstack([pc2, rgb])

    pts1 = pc1.reshape(pc1.shape[0] * pc1.shape[1], pc1.shape[2])
    pts2 = pc2.reshape(pc2.shape[0] * pc2.shape[1], pc2.shape[2])

    plotxyz(pts1, "r")
    plotxyzrgb(pts2, hold=True)
    import time

    time.sleep(4)
