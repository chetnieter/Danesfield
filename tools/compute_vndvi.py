#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Compute visible Normalized Difference Vegetation Index (vNDVI)
using RGB colored point cloud, basically following the approach in
Costa et al. A new visible band index (vNDVI) for estimating NDVI values on RGB images utilizing genetic algorithms. CEA 2020
'''

import argparse, logging, gdal, os, numpy as np
from danesfield.gdal_utils import gdal_open, gdal_save
from generate_rgb import main as computeRGB


def normalize(arr, rng): # return array scaled to range
    a, b = np.min(arr), np.max(arr)
    c, d = rng[0], rng[1]
    e, g = b-a, d-c
    if e:
        k = g/e    
        l = c - k*a
        res = k*arr + l
    else:
        res = .5*g*np.ones_like(arr)
    return res


def vNDVI(path2output): # return visible Normalized Difference Vegetation Index (vNDVI)
    def getBand(b): # return augmented color band
        FN = os.path.join(path2output, f'{b}.tif')
        img = gdal_open(FN)
        band = img.GetRasterBand(1)
        arr = band.ReadAsArray()
        m = arr[arr>0].min()
        arr[arr==band.GetNoDataValue()] = m # ensure some value
        arr[arr<=0] = m # to avoid division by 0
        return arr

    print('Generating vNDVI ...')
    R = getBand('R')
    G = getBand('G')
    B = getBand('B')
    # constants from the paper
    C, rp, gp, bp = 0.5268, -0.1294, 0.3389, -0.3118
    V = C * R**rp * G**gp * B**bp
    eps = 0.00075 # TODO param/config
    M = np.percentile(V, 100-eps)
    V[V>M] = M
    m = np.percentile(V, eps)
    V[V<m] = m
    res = normalize(V, [-1,1])
    return res


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', help='path/to/DSM.tif')
    parser.add_argument('input', help='path/to/colored/point/cloud.las')
    parser.add_argument('output', help='path/to/NDVI.tif')
    args = parser.parse_args(argv)
    
    imgFN = args.image
    path2output = os.path.dirname(args.output)
    computeRGB([path2output, '--source_points', args.input])
    vndvi = vNDVI(path2output)
    gdal_save(vndvi, gdal_open(imgFN), args.output,
              gdal.GDT_Float32,
              options=['COMPRESS=DEFLATE'])


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
