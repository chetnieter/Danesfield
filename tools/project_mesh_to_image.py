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

data_dir = Path('/home/local/KHQ/chet.nieter/data/BIG-R-T10/data/ElSegundo')
mesh_file = data_dir / 'elsegundo.ply'
envi_file = data_dir / 'chip_01.img'
rpc_file = data_dir / 'chip_01.img.rpc'
png_file = data_dir / 'chip_01.project.png'
lon_file = data_dir / 'chip_01_LonEnvi.img'
lat_file = data_dir / 'chip_01_LatEnvi.img'
hae_file = data_dir / 'chip_01_HaeEnvi.img'

offset = np.array([370332.4999829354, 3752535.9532070896, -19.399948061443865])

with open(mesh_file, 'rb') as f:
    plydata = PlyData.read(f)

with open(rpc_file, 'r') as f:
    rpc_model = raytheon_rpc.parse_raytheon_rpc_file(f)

orig_envi = gdal_utils.gdal_open(str(envi_file))

vertices = plydata['vertex']
vert_pts = np.stack([vertices.data['x'],
                     vertices.data['y'],
                     vertices.data['z']], axis=1) + offset

inProj = pyproj.Proj(proj='utm', zone=11, ellps='WGS84')
outProj = pyproj.Proj(proj='longlat', datum='WGS84')

lon, lat, hae = pyproj.transform(inProj, outProj, vert_pts[:,0], vert_pts[:,1], vert_pts[:,2])
wgs_coords = np.stack([lon, lat, hae], axis=1)

img_indices = rpc_model.project(wgs_coords)

h = 10639
w = 9973

img_data = np.zeros((h, w, 3), dtype=np.uint8)
lon_img = np.zeros((h, w), dtype=np.float64)
lat_img = np.zeros((h, w), dtype=np.float64)
hae_img = np.zeros((h, w), dtype=np.float64)

for arr_idx, img_idx in enumerate(img_indices):
    img_data[round(img_idx[1]), round(img_idx[0])] = np.ones(3)*255
    lon_img[round(img_idx[1]), round(img_idx[0])] = lon[arr_idx]
    lat_img[round(img_idx[1]), round(img_idx[0])] = lat[arr_idx]
    hae_img[round(img_idx[1]), round(img_idx[0])] = hae[arr_idx]

imsave(png_file, img_data)
gdal_utils.gdal_save(lon_img, orig_envi, str(lon_file), gdal.GDT_Float64)
gdal_utils.gdal_save(lat_img, orig_envi, str(lat_file), gdal.GDT_Float64)
gdal_utils.gdal_save(hae_img, orig_envi, str(hae_file), gdal.GDT_Float64)
