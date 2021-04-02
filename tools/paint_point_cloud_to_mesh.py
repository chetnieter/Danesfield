#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import argparse
import logging
import numpy
import pdal
import vtk
from vtk.util import numpy_support

def main(args):
    parser = argparse.ArgumentParser(
        description='Paints point cloud data on to a mesh')
    parser.add_argument("source_point_cloud", help="Source point cloud file name")
    parser.add_argument("source_mesh", help="Source mesh file name")
    parser.add_argument("destination_mesh", help="Destination mesh file name")

    args = parser.parse_args(args)

    # read the las point cloud file
    json = u"""
    {
      "pipeline": [
        {
            "type":"readers.las",
            "filename":"%s"
        }
      ]
    }"""
    print("Loading Point Cloud")
    json = json % args.source_point_cloud
    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good
    # this causes a segfault at the end of the program
    # pipeline.loglevel = 8  # really noisy
    pipeline.execute()
    points = pipeline.arrays[0]
    pipeline = None

    # read the mesh file
    print("Loading mesh")
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(args.source_mesh)
    obj_reader.Update()
    mesh = obj_reader.GetOutput()

    points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())

    # Shift the mesh
    # Hard-coding for now, make an input or parse from command line?
    utm_shift = [435516.726081, 3354093.8, -47.911346]

    for p in points:
      p += utm_shift

    # Create a color array
    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("Colors")
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(len(points))
    colors.FillComponent(0, 255)
    colors.FillComponent(1, 0)
    colors.FillComponent(2, 0)

    # mesh.GetPointData().SetScalars(colors);

    print("Writing new mesh")
    # obj_writer = vtk.vtkOBJWriter()
    # obj_writer.SetFileName(args.destination_mesh)
    # obj_writer.SetInputData(mesh)
    # obj_writer.Write()
    ply_writer = vtk.vtkPolyDataWriter()
    ply_writer.SetFileName(args.destination_mesh)
    ply_writer.SetInputData(mesh)
    ply_writer.Write()

if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)