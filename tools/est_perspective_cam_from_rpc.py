#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

# Projects mesh data to images via a RPC camera

import argparse
import numpy as np
import pyproj
from plyfile import PlyData, PlyElement
from danesfield import raytheon_rpc
from pathlib import Path

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
    tol = 1e-6
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

    proj[2, 3] = 1.0

    return proj

parser = argparse.ArgumentParser()
parser.add_argument('mesh_file', type=str, help='mesh file in ply format')
parser.add_argument('rpc_file', type=str, help='rpc camera in Raytheon format')
args = parser.parse_args()

offset = np.array([370332.4999829354, 3752535.9532070896, -19.399948061443865])

with open(args.mesh_file, 'rb') as f:
    plydata = PlyData.read(f)

with open(args.rpc_file, 'r') as f:
    rpc_model = raytheon_rpc.parse_raytheon_rpc_file(f)

vertices = plydata['vertex']
vert_pts = np.stack([vertices.data['x'],
                     vertices.data['y'],
                     vertices.data['z']], axis=1)
utm_pts = vert_pts + offset

inProj = pyproj.Proj(proj='utm', zone=11, ellps='WGS84')
outProj = pyproj.Proj(proj='longlat', datum='WGS84')

lon, lat, hae = pyproj.transform(inProj, outProj, utm_pts[:,0], utm_pts[:,1], utm_pts[:,2])
wgs_pts = np.stack([lon, lat, hae], axis=1)

img_pts = rpc_model.project(wgs_pts)

skip = 10
cam = compute_dlt(img_pts[::skip,:], vert_pts[::skip,:])

world_homo_pts = np.stack([vert_pts[:, 0],
                           vert_pts[:, 1],
                           vert_pts[:, 2],
                           np.ones(vert_pts.shape[0] )], axis=1)

img_homo_pts = (cam@world_homo_pts.T).T
new_img_pts = img_homo_pts[:,0:2]/img_homo_pts[:,2][:, np.newaxis]

rmse = np.sqrt(((new_img_pts - img_pts) ** 2).mean())

print('RMSE: ', rmse)

print(cam)
