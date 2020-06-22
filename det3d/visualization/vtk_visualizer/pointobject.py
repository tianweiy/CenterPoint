# -*- coding: utf-8 -*-
"""
VTK visualization object

@author: Øystein Skotheim, SINTEF ICT <oystein.skotheim@sintef.no>
"""

import vtk
from vtk.util import numpy_support
import numpy as np
from itertools import count


class VTKObject:
    """VTK visualization object
    Class that sets up the necessary VTK pipeline for displaying
    various objects (point clouds, meshes, geometric primitives)"""

    def __init__(self):
        self.verts = None
        self.cells = None
        self.scalars = None
        self.normals = None
        self.pd = None
        self.LUT = None
        self.mapper = None
        self.actor = None

    def CreateFromSTL(self, filename):
        "Create a visualization object from a given STL file"

        if not filename.lower().endswith(".stl"):
            raise Exception("Not an STL file")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(reader.GetOutput())

        self.SetupPipelineMesh()

    def CreateFromPLY(self, filename):
        "Create a visualization object from a given PLY file"

        if not filename.lower().endswith(".ply"):
            raise Exception("Not a PLY file")

        reader = vtk.vtkPLYReader()
        reader.SetFileName(filename)
        reader.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(reader.GetOutput())

        self.SetupPipelineMesh()

    def CreateFromActor(self, actor):
        "Create a visualization object from a given vtkActor instance"
        if not isinstance(actor, vtk.vtkObject):
            raise Exception("Argument is not a VTK Object")
        self.actor = actor

    def CreateFromPolyData(self, pd):
        "Create a visualization object from a given vtkPolyData instance"
        self.pd = pd
        self.SetupPipelineMesh()

    def CreateFromArray(self, pc):
        """Create a point cloud visualization object from a given NumPy array
        
        The NumPy array should have dimension Nxd where d >= 3
        
        If d>3, the points will be colored according to the last column
        in the supplied array (values should be between 0 and 1, where 
        0 is black and 1 is white)
        """

        nCoords = pc.shape[0]
        nElem = pc.shape[1]

        if nElem < 3:
            raise Exception("Number of elements must be greater than or equal to 3")

        self.verts = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.scalars = None
        self.pd = vtk.vtkPolyData()

        # Optimized version of creating the vertices array
        # - We need to be sure to keep the supplied NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        self.points_npy = pc[:, :3].copy()
        self.verts.SetData(numpy_support.numpy_to_vtk(self.points_npy))

        # Optimized version of creating the cell ID array
        # - We need to be sure to keep the supplied NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - Note that the cell array looks like this: [1 vtx0 1 vtx1 1 vtx2 1 vtx3 ... ]
        #   because it consists of vertices with one primitive per cell
        self.cells_npy = np.vstack(
            [np.ones(nCoords, dtype=np.int64), np.arange(nCoords, dtype=np.int64)]
        ).T.flatten()

        self.cells.SetCells(
            nCoords, numpy_support.numpy_to_vtkIdTypeArray(self.cells_npy)
        )

        self.pd.SetPoints(self.verts)
        self.pd.SetVerts(self.cells)

        # Optimized version of creating the scalars array
        # - We need to be sure to keep the NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        if nElem > 3:
            self.scalars_npy = pc[:, -1].copy()
            self.scalars = numpy_support.numpy_to_vtk(self.scalars_npy)

            # Color the scalar data in gray values
            self.LUT = vtk.vtkLookupTable()
            self.LUT.SetNumberOfColors(255)
            self.LUT.SetSaturationRange(0, 0)
            self.LUT.SetHueRange(0, 0)
            self.LUT.SetValueRange(0, 1)
            self.LUT.Build()
            self.scalars.SetLookupTable(self.LUT)
            self.pd.GetPointData().SetScalars(self.scalars)

        self.SetupPipelineCloud()

    def AddNormals(self, ndata):
        "Add surface normals (Nx3 NumPy array) to the current point cloud visualization object"

        nNormals = ndata.shape[0]
        nDim = ndata.shape[1]

        if nDim != 3:
            raise Exception("Expected Nx3 array of surface normals")

        if self.pd.GetNumberOfPoints() == 0:
            raise Exception("No points to add normals for")

        if nNormals != self.pd.GetNumberOfPoints():
            raise Exception(
                "Supplied number of normals incompatible with number of points"
            )

        # Optimized version of creating the scalars array
        # - We need to be sure to keep the NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        self.normals_npy = ndata.copy()
        self.normals = numpy_support.numpy_to_vtk(self.normals_npy)

        self.pd.GetPointData().SetNormals(self.normals)
        self.pd.Modified()

    def AddColors(self, cdata):
        """"Add colors (Nx3 NumPy array) to the current point cloud visualization object        
        NumPy array should be of type uint8 with R, G and B values between 0 and 255
        """

        nColors = cdata.shape[0]
        nDim = cdata.shape[1]

        if self.pd.GetNumberOfPoints() == 0:
            raise Exception("No points to add color for")

        if nColors != self.pd.GetNumberOfPoints():
            raise Exception(
                "Supplied number of colors incompatible with number of points"
            )

        if nDim != 3:
            raise Exception("Expected Nx3 array of colors")

        # Optimized version of creating the scalars array
        # - We need to be sure to keep the NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        self.colors_npy = cdata.copy().astype(np.uint8)
        self.colors = numpy_support.numpy_to_vtk(self.colors_npy)

        self.pd.GetPointData().SetScalars(self.colors)
        self.pd.Modified()

    def SetupPipelineCloud(self):
        "Set up the VTK pipeline for visualizing a point cloud"

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)

        if self.scalars != None:
            self.ScalarsOn()
            self.SetScalarRange(0, 1)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToPoints()
        self.actor.GetProperty().SetColor(0.0, 1.0, 0.0)

    def SetupPipelineMesh(self):
        "Set up the VTK pipeline for visualizing a mesh"

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)

        if self.scalars != None:
            if self.pd.GetPointData() != None:
                if (self.pd.GetPointData().GetScalars != None) and (self.LUT != None):
                    self.pd.GetPointData().GetScalars().SetLookupTable(self.LUT)

            self.mapper.ScalarVisibilityOn()
            self.mapper.SetColorModeToMapScalars()
            self.mapper.SetScalarModeToUsePointData()
            self.mapper.SetScalarRange(0, 1)

        else:
            self.mapper.ScalarVisibilityOff()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToSurface()
        self.actor.GetProperty().SetInterpolationToGouraud()

    def SetupPipelineHedgeHog(self, scale=15.0):
        """Set up the VTK pipeline for visualizing points with surface normals"
        
        The surface normals are visualized as lines with the given scale"""

        hh = vtk.vtkHedgeHog()
        hh.SetInputData(self.pd)
        hh.SetVectorModeToUseNormal()
        hh.SetScaleFactor(scale)
        hh.Update()

        # Set up mappers and actors
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(hh.GetOutputPort())
        self.mapper.Update()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(1, 0, 0)  # Default color

    def ScalarsOn(self):
        "Enable coloring of the points based on scalar array"
        self.mapper.ScalarVisibilityOn()
        self.mapper.SetColorModeToMapScalars()
        self.mapper.SetScalarModeToUsePointData()

    def ScalarsOff(self):
        "Disable coloring of the points based on scalar array"
        self.mapper.ScalarVisibilityOff()

    def SetScalarRange(self, smin, smax):
        "Set the minimum and maximum values for the scalar array"
        self.mapper.SetScalarRange(smin, smax)

    def GetActor(self):
        "Returns the current actor"
        return self.actor

    # Primitives
    def CreateSphere(self, origin, r):
        "Create a sphere with given origin (x,y,z) and radius r"
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(origin)
        sphere.SetRadius(r)
        sphere.SetPhiResolution(25)
        sphere.SetThetaResolution(25)
        sphere.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(sphere.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateCylinder(self, origin, r, h):
        "Create a cylinder with given origin (x,y,z), radius r and height h"
        cyl = vtk.vtkCylinderSource()
        cyl.SetCenter(origin)
        cyl.SetRadius(r)
        cyl.SetHeight(h)
        cyl.SetResolution(25)
        cyl.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(cyl.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateAxes(self, length):
        "Create a coordinate axes system with a given length of the axes"
        axesActor = vtk.vtkAxesActor()
        axesActor.AxisLabelsOff()
        axesActor.SetTotalLength(length, length, length)
        self.actor = axesActor

    def CreatePlane(self, normal=None, origin=None):
        "Create a plane (optionally with a given normal vector and origin)"
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(10)
        plane.SetYResolution(10)
        if not normal is None:
            plane.SetNormal(normal)
        if not origin is None:
            plane.SetCenter(origin)
        plane.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(plane.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateBox(self, bounds):
        "Create a box with the given bounds [xmin,xmax,ymin,ymax,zmin,zmax]"
        box = vtk.vtkTessellatedBoxSource()
        box.SetBounds(bounds)
        box.Update()
        self.pd = box.GetOutput()
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateLine(self, p1, p2):
        "Create a 3D line from p1=[x1,y1,z1] to p2=[x2,y2,z2]"
        line = vtk.vtkLineSource()
        line.SetPoint1(*p1)
        line.SetPoint2(*p2)
        line.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(line.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateTriangles(self, pc, triangles):
        "Create a mesh from triangles"

        self.verts = vtk.vtkPoints()
        self.points_npy = pc[:, :3].copy()
        self.verts.SetData(numpy_support.numpy_to_vtk(self.points_npy))

        nTri = len(triangles)
        self.cells = vtk.vtkCellArray()
        self.pd = vtk.vtkPolyData()
        # - Note that the cell array looks like this: [3 vtx0 vtx1 vtx2 3 vtx3 ... ]
        self.cells_npy = np.column_stack(
            [np.full(nTri, 3, dtype=np.int64), triangles.astype(np.int64)]
        ).ravel()
        self.cells.SetCells(nTri, numpy_support.numpy_to_vtkIdTypeArray(self.cells_npy))

        self.pd.SetPoints(self.verts)
        self.pd.SetPolys(self.cells)

        self.SetupPipelineMesh()

    def CreatePolyLine(self, points):
        "Create a 3D line from Nx3 numpy array"
        self.verts = vtk.vtkPoints()
        polyline = vtk.vtkPolyLine()

        polyline_pid = polyline.GetPointIds()
        for i, p in enumerate(points):
            self.verts.InsertNextPoint(*tuple(p))
            polyline_pid.InsertNextId(i)

        polyline_cell = vtk.vtkCellArray()
        polyline_cell.InsertNextCell(polyline)

        self.pd = vtk.vtkPolyData()
        self.pd.SetPoints(self.verts)
        self.pd.SetLines(polyline_cell)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetLineStipplePattern(0xF0F0)
        self.actor.GetProperty().SetLineStippleRepeatFactor(1)
        self.actor.GetProperty().SetLineWidth(1.5)

    def CreatePolygon(self, points):
        "Create a 3D plane from Nx3 numpy array"
        self.verts = vtk.vtkPoints()
        polygon = vtk.vtkPolygon()

        polygon_pid = polygon.GetPointIds()
        for i, p in enumerate(points):
            self.verts.InsertNextPoint(*tuple(p))
            polygon_pid.InsertNextId(i)

        polygon_cell = vtk.vtkCellArray()
        polygon_cell.InsertNextCell(polygon)

        self.pd = vtk.vtkPolyData()
        self.pd.SetPoints(self.verts)
        self.pd.SetPolys(polygon_cell)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

    def CreateAxesSimplified(self, length):
        "Create a simplified coordinate axes system with 3 lines"
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        ptIds = [None, None]
        end_points = length * np.eye(3)
        line_colors = 255 * np.eye(3, dtype="i")
        for i in range(3):
            ptIds[0] = points.InsertNextPoint([0, 0, 0])
            ptIds[1] = points.InsertNextPoint(end_points[i])
            lines.InsertNextCell(2, ptIds)

            colors.InsertNextTuple3(*line_colors[i])
            colors.InsertNextTuple3(*line_colors[i])

        # Add the lines to the polydata container
        self.pd = vtk.vtkPolyData()
        self.pd.SetPoints(points)
        self.pd.GetPointData().SetScalars(colors)
        self.pd.SetLines(lines)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)
        self.mapper.Update()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

    def AddPoses(self, matrix_list):
        """"Add poses (4x4 NumPy arrays) to the object 
        """
        R_list = [p[:3, :3] for p in matrix_list]  # translation data
        t_list = [p[:3, 3] for p in matrix_list]  # rotation data

        self.points = vtk.vtkPoints()  # where t goes
        self.tensors = vtk.vtkDoubleArray()  # where R goes, column major
        self.tensors.SetNumberOfComponents(9)
        for i, R, t in zip(count(), R_list, t_list):
            self.points.InsertNextPoint(*tuple(t))
            self.tensors.InsertNextTypedTuple(tuple(R.ravel(order="F")))

        self.pose_pd = vtk.vtkPolyData()
        self.pose_pd.SetPoints(self.points)
        self.pose_pd.GetPointData().SetTensors(self.tensors)

    def SetupPipelinePose(self):
        """Set up the VTK pipeline for visualizing multiple copies of a geometric 
        representation with different poses"""

        # use vtkTensorGlyph set 3d pose for each actor instance
        tensorGlyph = vtk.vtkTensorGlyph()
        tensorGlyph.SetInputData(self.pose_pd)  # setup with AddPoses()
        tensorGlyph.SetSourceData(self.pd)
        tensorGlyph.ColorGlyphsOff()
        tensorGlyph.ThreeGlyphsOff()
        tensorGlyph.ExtractEigenvaluesOff()
        tensorGlyph.Update()

        # Set up mappers and actors
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(tensorGlyph.GetOutput())
        self.mapper.Update()
        self.actor = vtk.vtkActor()
        self.actor.GetProperty().SetLineWidth(1.5)
        self.actor.SetMapper(self.mapper)


if __name__ == "__main__":

    # Create some random points
    pc = np.random.rand(10000, 3)

    # Create some random normals
    normals = np.random.rand(10000, 3) - 0.5

    # Create some random colors
    colors = np.random.rand(10000, 3)
    colors *= 255.0
    colors = colors.astype(np.uint8)

    obj = VTKObject()
    obj.CreateFromArray(pc)
    obj.AddColors(colors)
    obj.AddNormals(normals)
    obj.SetupPipelineHedgeHog(0.2)

    ren = vtk.vtkRenderer()
    ren.AddActor(obj.GetActor())

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()
