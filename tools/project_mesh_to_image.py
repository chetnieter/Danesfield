#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

# Projects mesh data to images via a RPC camera

import numpy as np
from plyfile import PlyData, PlyElement
from danesfield import raytheon_rpc

mesh_file = '/home/local/KHQ/chet.nieter/data/BIG-R-T10/data/ElSegundo/elsegundo.ply'
rpc_file = '/home/local/KHQ/chet.nieter/data/BIG-R-T10/data/ElSegundo/chip_01.img.rpc'

offset = np.array([-3950779.358208671, 12578441.244892504, 4836.518238477409])

with open(mesh_file, 'rb') as f:
    plydata = PlyData.read(f)

vertices = plydata['vertex']
print(vertices.data['x'].shape)

num_points = 50

vert_pts = np.stack([vertices.data['x'][:num_points],
                     vertices.data['y'][:num_points],
                     vertices.data['z'][:num_points]], axis=1) + offset
print(vert_pts.shape)
print(offset.shape)
print(vert_pts)

rpc_model = raytheon_rpc.parse_raytheon_rpc_file(rpc_file)

first_point = vert_pts[0]
test_point = np.array([-118.3922626123161, 33.91261515519195, 34])

test_index = rpc_model.project(test_point)

print(test_point)
print(test_index)
