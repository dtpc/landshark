"""Utilities."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np

from landshark.basetypes import (CategoricalType, ContinuousType,
                                 CoordinateType, MissingType)

log = logging.getLogger(__name__)


def to_masked(array: np.ndarray,
              missing_value: MissingType
              ) -> np.ma.MaskedArray:
    """Create a masked array from array plus list of missing."""
    if missing_value is None:
        marray = np.ma.MaskedArray(data=array, mask=np.ma.nomask)
    else:
        mask = array == missing_value
        marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


def mb_to_points(batchMB: float,
                 ndim_con: int,
                 ndim_cat: int,
                 ndim_coord: int = 0,
                 halfwidth: int = 0
                 ) -> int:
    log.info("Batch size of {}MB requested".format(batchMB))
    patchsize = (halfwidth * 2 + 1) ** 2
    bytes_con = np.dtype(ContinuousType).itemsize * ndim_con
    bytes_cat = np.dtype(CategoricalType).itemsize * ndim_cat
    bytes_coord = np.dtype(CoordinateType).itemsize * ndim_coord
    mbytes_per_point = (bytes_con + bytes_cat + bytes_coord) * patchsize * 1e-6
    npoints = int(round(max(1., batchMB / mbytes_per_point)))
    log.info("Batch size set to {} points, total {:0.2f}MB".format(
        npoints, npoints * mbytes_per_point))
    return npoints


def mb_to_rows(batchMB: float,
               row_width: int,
               ndim_con: int,
               ndim_cat: int,
               halfwidth: int = 0
               ) -> int:
    log.info("Batch size of {}MB requested".format(batchMB))
    patchsize = (halfwidth * 2 + 1) ** 2
    bytes_con = np.dtype(ContinuousType).itemsize * ndim_con
    bytes_cat = np.dtype(CategoricalType).itemsize * ndim_cat
    point_mbytes = (bytes_con + bytes_cat) * patchsize * 1e-6
    npoints = batchMB / point_mbytes
    nrows = int(round(max(1., npoints / row_width)))
    log.info("Batch size set to {} rows, total {:0.2f}MB".format(
        nrows, point_mbytes * row_width * nrows))
    return nrows
