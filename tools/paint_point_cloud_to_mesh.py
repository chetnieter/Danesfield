#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import argparse
import logging
import numpy as np
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

    # Read input OBJ header
    with open(args.source_mesh, "r") as in_f:
        lines = in_f.readlines()
    if len(lines) > 0:
        i = 0
        while i < len(lines) and len(lines[i]) > 0 and lines[i][0] == "#":
            i += 1
    else:
        print("Warning: empty input file.")
        sys.exit(0)
    header = lines[:i]

    # Set the shift values for the mesh from header data
    utm_shift = np.zeros(3)
    for l in header:
      cols = l.split()
      if '#x' in cols[0]:
          utm_shift[0] = float(cols[2])
      elif '#y' in cols[0]:
          utm_shift[1] = float(cols[2])
      elif '#z' in cols[0]:
          utm_shift[2] = float(cols[2])

    print(utm_shift)

    # read the mesh file
    print("Loading mesh")
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(args.source_mesh)
    obj_reader.Update()
    mesh = obj_reader.GetOutput()

    vertices = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())

    # Shift the mesh
    for v in vertices:
        v += utm_shift

    # Get the bounding box of the mesh
    mesh.ComputeBounds()
    bbox = mesh.GetBounds()

    # read the las point cloud file, filtering on the mesh bounding box
    pdal_input = u"""
    [
      "{}",
      {{
        "type":"filters.range",
        "limits":"X[{}:{}], Y[{}:{}], Z[{}:{}]"
      }}
    ]"""
    print("Loading Point Cloud")
    pdal_input = pdal_input.format(args.source_point_cloud,
                                   bbox[0], bbox[1],
                                   bbox[2], bbox[3],
                                   bbox[4], bbox[5])
    pipeline = pdal.Pipeline(pdal_input)
    pipeline.validate()  # check if our JSON and options were good
    # this causes a segfault at the end of the program
    # pipeline.loglevel = 8  # really noisy
    pipeline.execute()
    points = pipeline.arrays[0]
    pipeline = None

    # Array to store the vertex colors
    vertex_colors = np.zeros( (len(vertices), 3) ,dtype=np.int32)
    vertex_num_points = np.zeros( (len(vertices) ) )

    # Loop over point cloud and assign color to closest vertex
    for p in points:
      shortest_distance = np.finfo(np.float32).max
      shortest_idx = 0

      pt = np.array([p[0], p[1], p[2]])
      sep_vect = vertices - pt
      dist_vect = np.einsum('ij,ij->i', sep_vect, sep_vect)

      shortest_idx = np.argmin(dist_vect)

      vertex_colors[shortest_idx, :] = np.array([p[12], p[13], p[14]])
      vertex_num_points[shortest_idx] += 1

    for i in range(len(vertex_colors)):
      if vertex_num_points[i] > 0:
        vertex_colors[i] = vertex_colors[i]/vertex_num_points[i]

    # Create a color array
    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("Colors")
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(len(vertices))

    for i in range(len(vertices)):
      colors.SetComponent(i, 0, vertex_colors[i, 0])
      colors.SetComponent(i, 1, vertex_colors[i, 1])
      colors.SetComponent(i, 2, vertex_colors[i, 2])

    mesh.GetPointData().SetScalars(colors);

    print("Writing new mesh")
    mesh_writer = vtk.vtkPLYWriter()
    mesh_writer.SetArrayName("Colors")
    mesh_writer.SetFileName(args.destination_mesh)
    mesh_writer.SetInputData(mesh)
    mesh_writer.Write()

if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)