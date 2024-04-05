#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

# Projects mesh data to images via a RPC camera

import gdal
import numpy as np
import pyproj
from plyfile import PlyData, PlyElement
from danesfield import gdal_utils, raytheon_rpc
from pathlib import Path
from skimage.io import imsave

# Function to estimate camera from set of image and 3d points. Based on VXL
# implementation
def compute_dlt(image_pts, world_pts):

    cam = np.zeros( (3,4) )

    if image_pts.shape[0] < 6:
        print('Must have 6 or more points to estimate camera')
        return cam
    if image_pts.shape[0] != world_pts.shape[0]:
        print('Must have the same number of image and world points')
        return cam

    # Two equations for each point, one for the x's, the other for the ys
    num_eqns = 2*image_pts.shape[0]

    # A 3x4 projection matrix has 11 free vars
    num_vars = 11

    A = np.zeros( (num_eqns, num_vars) )
    b = np.zeros( (num_eqns, 1) )

    A[0::2, :3] = world_pts[:,:]
    A[0::2, 3] = 1.0
    A[0::2, 8:11] = -np.multiply(world_pts, image_pts[:, 0][:, np.newaxis])

    A[1::2, 4:7] = world_pts[:,:]
    A[1::2, 7] = 1.0
    A[1::2, 8:11] = -np.multiply(world_pts, image_pts[:, 1][:, np.newaxis])

    b[0::2, 0] = image_pts[:, 0]
    b[1::2, 0] = image_pts[:, 1]

    # Solve for x using SVD
    U, S, Vt = np.linalg.svd(A)
    tol = 1e-3
    r = np.sum(S > tol)
    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]
    # Compute the inverse of S
    Si = np.diag(1 / S)
    x = Vt.T @ Si @ U.T @ b

    proj = np.zeros( (3, 4) )

    for row in range(3):
        for col in range(4):
            if row * 4 + col < 11:
                proj[row, col] = x[row * 4 + col]

    print(proj)

    return cam

data_dir = Path('/home/local/KHQ/chet.nieter/data/BIG-R-T10/data/ElSegundo')
mesh_file = data_dir / 'elsegundo.ply'
rpc_file = data_dir / 'chip_01.img.rpc'

offset = np.array([370332.4999829354, 3752535.9532070896, -19.399948061443865])

with open(mesh_file, 'rb') as f:
    plydata = PlyData.read(f)

with open(rpc_file, 'r') as f:
    rpc_model = raytheon_rpc.parse_raytheon_rpc_file(f)

vertices = plydata['vertex']
world_pts = np.stack([vertices.data['x'],
                     vertices.data['y'],
                     vertices.data['z']], axis=1) + offset

inProj = pyproj.Proj(proj='utm', zone=11, ellps='WGS84')
outProj = pyproj.Proj(proj='longlat', datum='WGS84')

lon, lat, hae = pyproj.transform(inProj, outProj, world_pts[:,0], world_pts[:,1], world_pts[:,2])
wgs_coords = np.stack([lon, lat, hae], axis=1)

img_pts = rpc_model.project(wgs_coords)

num_points = 100
cam = compute_dlt(img_pts[:num_points,:], wgs_coords[:num_points,:])

print(cam)
