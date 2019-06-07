"""Test kfold module."""

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

from itertools import groupby
from operator import itemgetter

import numpy as np
import pytest

from landshark.kfold import BlockedKFolder, KFolder

fold_params = [
    (10, 2, 5, 10),
    (123456, 10, 99, 1000)
]


@pytest.mark.parametrize("N,K,B,W", fold_params)
def test_kfolder(N, K, B, W):
    folder = KFolder(K)
    batches = [B] * (N // B)
    if N % B > 0:
        batches.append(N % B)
    ixs = [
        np.vstack(
            np.unravel_index(np.random.randint(W ** 2, size=b), (W, W))
        ).T for b in batches
    ]
    folds = [folder(ix) for ix in ixs]
    bs = [len(b) for b in folds]
    assert bs == [B] * (N // B) + [] if N % B == 0 else [N % B]
    folds_flat = [i for b in folds for i in b]
    assert len(set(folds_flat)) == K
    assert min(folds_flat) > 0
    assert max(folds_flat) <= K
    assert set(folder.counts.keys()) == set(range(1, K + 1))
    assert sum(folder.counts.values()) == N


blk_size_params = [[2, 5], [13, 103]]
blk_fold_params = [
    (*fp, b) for fp, bp in zip(fold_params, blk_size_params) for b in bp
]


@pytest.mark.parametrize("N,K,B,W,block_size_px", blk_fold_params)
def test_blocked_kfolder(N, K, B, W, block_size_px):
    folder = BlockedKFolder((W, W), block_size_px, K)
    batches = [B] * (N // B)
    if N % B > 0:
        batches.append(N % B)
    ixs = [
        np.vstack(
            np.unravel_index(np.random.randint(W ** 2, size=b), (W, W))
        ).T for b in batches
    ]
    folds = [folder(ix) for ix in ixs]
    bs = [len(b) for b in folds]
    assert bs == [B] * (N // B) + [] if N % B == 0 else [N % B]
    folds_flat = [i for b in folds for i in b]
    assert len(set(folds_flat)) == K
    assert min(folds_flat) > 0
    assert max(folds_flat) <= K
    assert set(folder.counts.keys()) == set(range(1, K + 1))
    assert sum(folder.counts.values()) == N
    blk_ixs = [
        (i // block_size_px, j // block_size_px) for i, j in np.vstack(ixs)
    ]
    blk_grps = groupby(sorted(zip(blk_ixs, folds_flat)), itemgetter(0))
    blk_folds = {k: set(v[1] for v in vals) for k, vals in blk_grps}
    print(blk_folds)
    assert max(map(len, blk_folds.values()))
