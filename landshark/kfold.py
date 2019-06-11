"""Cross validation indices."""

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

from typing import Tuple

import numpy as np


class KFolder:
    """Generate random k-fold indices from training data."""

    def __init__(self, K: int = 10, random_seed: int = 220) -> None:
        self.K = K
        self.rnd = np.random.RandomState(random_seed)
        self.counts = {k: 0 for k in range(1, self.K + 1)}

    def generate_folds(self, indexes: np.ndarray) -> np.ndarray:
        """Randomly generate the fold number for each training point."""
        batch_n = indexes.shape[0]
        folds = self.rnd.randint(1, self.K + 1, size=batch_n)
        return folds

    def __call__(self, indexes: np.ndarray) -> np.ndarray:
        """Generate folds numbers and record counts."""
        folds = self.generate_folds(indexes)
        indices, counts = np.unique(folds, return_counts=True)
        for k, v in zip(indices, counts):
            self.counts[k] += v
        return folds


class BlockedKFolder(KFolder):
    """Generate random k-fold indices grouped together by image block."""

    def __init__(
        self,
        im_shape: Tuple[int, int],
        block_size_px: int = 100,
        K: int = 10,
        random_seed: int = 220,
    ) -> None:
        super().__init__(K=K, random_seed=random_seed)
        self.block_size_px = block_size_px
        self.im_shape = im_shape
        self.im_shape_blk = (
            int(np.ceil(self.im_shape[0] / self.block_size_px)),
            int(np.ceil(self.im_shape[1] / self.block_size_px))
        )
        self.n_blocks = np.prod(self.im_shape_blk)
        self.block_folds = self.rnd.randint(
            1, K + 1, size=self.n_blocks, dtype=np.uint8
        )

    def generate_folds(self, indexes: np.ndarray) -> np.ndarray:
        """Generate folds grouped by image blocks."""
        block_ixs = indexes // self.block_size_px
        block_ids = np.ravel_multi_index(block_ixs.T, self.im_shape_blk)
        folds = self.block_folds[block_ids]
        return folds
