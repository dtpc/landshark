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

from typing import Set, Tuple

import numpy as np


class KFolder:
    """Generate random k-fold indices from training data."""

    def __init__(self, K: int = 10, random_seed: int = 220) -> None:
        assert K > 1
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
        self.im_shape = im_shape
        self.block_size_px = block_size_px
        self.K = K
        self._gen_rand_blk_folds_array()

    def _gen_rand_blk_folds_array(self) -> None:
        rows = int(np.ceil(self.im_shape[0] / self.block_size_px))
        cols = int(np.ceil(self.im_shape[1] / self.block_size_px))
        folds = np.full((rows, cols), 0)
        for i in range(rows):
            for j in range(cols):
                excl: Set[int] = {
                    0,
                    folds[i - 1, j] if i > 0 else 0,
                    folds[i, j - 1] if j > 0 else 0,
                }
                f = 0
                while f in excl:
                    f = self.rnd.randint(1, self.K + 1)
                folds[i, j] = f
        self.block_folds = folds

    def generate_folds(self, indexes: np.ndarray) -> np.ndarray:
        """Generate folds grouped by image blocks."""
        block_ixs = indexes // self.block_size_px
        folds = self.block_folds[block_ixs[:, 0], block_ixs[:, 1]]
        return folds
