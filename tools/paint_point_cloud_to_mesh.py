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
    json = u"""
    {
      "pipeline": [
        {
            "type":"readers.obj",
            "filename":"%s"
        },
        {
          "type":"writers.ply",
          "storage_mode":"little endian",
          "filename":"{}"
        }
      ]
    }"""
    print("Loading Mesh")
    json = json % args.source_mesh
    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good
    # this causes a segfault at the end of the program
    # pipeline.loglevel = 8  # really noisy
    pipeline.execute()
    mesh = pipeline.arrays[0]
    pipeline = None

    # Shift the mesh
    # Hard-coding for now, make an input or parse from command line?
    utm_shift = [435516.726081, 3354093.8, -47.911346]

    for m in mesh:
      m[0] += utm_shift[0]
      m[1] += utm_shift[1]
      m[2] += utm_shift[2]

    # Write out mesh
    json = u"""
    {
      "pipeline": [
        {
          "type":"writers.ply",
          "storage_mode":"little endian",
          "filename":"%s"
        }
      ]
    }"""
    print("Writing Mesh")
    json = json % args.destination_mesh
    pipeline = pdal.Pipeline(json, [mesh])
    pipeline.validate()  # check if our JSON and options were good
    pipeline.execute()
    pipeline = None

if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)