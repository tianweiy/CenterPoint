# -*- coding: utf-8 -*-
"""
Render widget for VTK based on Qt

:Author: Øystein Skotheim, Zivid Labs <oystein.skotheim@zividlabs.com>
"""

import sys

import vtk

# from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# Note: The QVTKRenderWindowInteractor included with VTK is buggy for PySide
# Use our modified version instead
from vtk_visualizer.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

qt_api = "pyqt"
try:
    from PyQt5.QtWidgets import *
except:
    qt_api = "pyside"

if qt_api == "pyside":
    try:
        from PySide import QtGui
    except:
        raise Exception("Need either PySide or PyQt4")


class RenderWidget:
    def __init__(self, renderer=None, parent=None):

        # Every QT app needs a QApplication
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Create the widget
        if renderer is None:
            self.renderer = vtk.vtkRenderer()
        else:
            self.renderer = renderer

        self.widget = QVTKRenderWindowInteractor(parent)

        self.widget.Initialize
        self.widget.Start()

        # Set the interactor style
        self.widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # Get the render window from the widget
        self.renderWindow = self.widget.GetRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)

        # show the widget
        self.widget.show()

    def exec_(self):
        self.widget.show()  # make sure we are shown
        self.app.exec_()


if __name__ == "__main__":

    cone = vtk.vtkConeSource()
    cone.SetResolution(8)

    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(coneActor)

    w = RenderWidget(ren)
    w.exec_()
