import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

NUM_COLORS = 10
IGNORE_TOKEN_ID = 10
VOCAB_SIZE = 11


def _grid_fits(grid: List[List[int]], grid_size: int) -> bool:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h > 0 and w > 0 and h <= grid_size and w <= grid_size


def _pad_grid(grid: List[List[int]], grid_size: int) -> np.ndarray:
    arr = np.full((grid_size, grid_size), IGNORE_TOKEN_ID, dtype=np.int32)
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return arr
    arr[:h, :w] = np.array(grid, dtype=np.int32)
    return arr


def _example_fits(example: dict, grid_size: int) -> bool:
    inp = example.get("input")
    out = example.get("output")
    if inp is None or out is None:
        return False
    return _grid_fits(inp, grid_size) and _grid_fits(out, grid_size)


def _build_positions(num_support: int, grid_size: int, patch_size: int) -> np.ndarray:
    if grid_size % patch_size != 0:
        raise ValueError("grid_size must be divisible by patch_size")
    grid = grid_size // patch_size
    num_grids = 2 * (num_support + 1)
    tokens_per_grid = grid * grid
    positions = np.zeros((num_grids * tokens_per_grid, 4), dtype=np.int32)
    idx = 0
    for grid_idx in range(num_grids):
        example_idx = grid_idx // 2
        io_flag = grid_idx % 2
        for py in range(grid):
            for px in range(grid):
                positions[idx] = (io_flag, px, py, example_idx)
                idx += 1
    return positions


def _build_attention_mask(
    grids: np.ndarray, grid_size: int, patch_size: int
) -> np.ndarray:
    grid = grid_size // patch_size
    bsz, num_grids, _, _ = grids.shape
    patches = grids.reshape(
        bsz,
        num_grids,
        grid,
        patch_size,
        grid,
        patch_size,
    )
    patches = np.transpose(patches, (0, 1, 2, 4, 3, 5))
    patch_all_ignore = np.all(patches == IGNORE_TOKEN_ID, axis=(4, 5))
    patch_mask = ~patch_all_ignore
    patch_mask[:, -1, :] = True
    return patch_mask.reshape(bsz, num_grids * grid * grid)


class Dataset:
    def __init__(
        self,
        path: Path,
        split: str,
        subset: str = "train",
        *,
        extra_train_path: Optional[Path] = None,
        seed: int = 0,
        batch_size: int = 256,
        shuffle: bool = True,
        num_support: int = 4,
        grid_size: int = 32,
        patch_size: int = 4,
    ) -> None:
        self.path = Path(path)
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.num_support = num_support
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.tasks_by_name = {}
        self._load_dir(self.path / split, merge_existing=False)
        if self.subset == "train" and extra_train_path is not None:
            self._load_dir(Path(extra_train_path) / split, merge_existing=True)

        self.tasks = [self.tasks_by_name[name] for name in sorted(self.tasks_by_name)]
        self.num_tasks = len(self.tasks)
        self.query_pairs = self._build_query_pairs()
        self.num_samples = len(self.query_pairs)

        self.positions_template = _build_positions(num_support, grid_size, patch_size)

    def _load_dir(self, directory: Path, *, merge_existing: bool) -> None:
        files = sorted(directory.glob("*.json"))
        for file_path in files:
            task_name = file_path.stem
            with file_path.open("r") as fh:
                task_data = json.load(fh)
            train_examples = [
                ex
                for ex in task_data.get("train", [])
                if _example_fits(ex, self.grid_size)
            ]
            test_examples = [
                ex
                for ex in task_data.get("test", [])
                if _example_fits(ex, self.grid_size)
            ]
            if not train_examples:
                continue
            if merge_existing:
                if task_name not in self.tasks_by_name:
                    continue
                self.tasks_by_name[task_name]["train"].extend(train_examples)
                continue
            self.tasks_by_name[task_name] = {
                "train": list(train_examples),
                "test": list(test_examples),
            }

    def _build_query_pairs(self) -> List[Tuple[int, int]]:
        query_pairs = []
        for task_idx, task in enumerate(self.tasks):
            examples_key = "train" if self.subset == "train" else "test"
            query_examples = task.get(examples_key, [])
            for query_idx in range(len(query_examples)):
                query_pairs.append((task_idx, query_idx))
        return query_pairs

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def _sample_support_indices(self, num_train: int, query_idx: int) -> np.ndarray:
        if self.num_support == 0:
            return np.zeros((0,), dtype=np.int32)
        if num_train <= 0:
            return np.zeros((self.num_support,), dtype=np.int32)

        if self.subset == "train" and num_train > 1:
            candidates = [i for i in range(num_train) if i != query_idx]
        else:
            candidates = list(range(num_train))

        if not candidates:
            candidates = list(range(num_train))

        replace = len(candidates) < self.num_support
        return self.rng.choice(candidates, size=self.num_support, replace=replace)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        usable = self.num_samples - (self.num_samples % self.batch_size)

        for start in range(0, usable, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            grids_batch = []

            for pair_idx in batch_idx:
                task_idx, query_idx = self.query_pairs[pair_idx]
                task = self.tasks[task_idx]
                train_examples = task["train"]
                query_examples = (
                    task["train"] if self.subset == "train" else task["test"]
                )
                query = query_examples[query_idx]

                support_indices = self._sample_support_indices(
                    len(train_examples), query_idx
                )

                grids = []
                for support_idx in support_indices:
                    support = train_examples[support_idx]
                    grids.append(_pad_grid(support["input"], self.grid_size))
                    grids.append(_pad_grid(support["output"], self.grid_size))

                grids.append(_pad_grid(query["input"], self.grid_size))
                grids.append(_pad_grid(query["output"], self.grid_size))

                grids_batch.append(np.stack(grids, axis=0))

            grids_arr = np.stack(grids_batch, axis=0).astype(np.int32)
            batch_size = grids_arr.shape[0]

            attention_mask = _build_attention_mask(
                grids_arr, self.grid_size, self.patch_size
            )
            yield {
                "grids": grids_arr,
                "positions": np.broadcast_to(
                    self.positions_template,
                    (batch_size, self.positions_template.shape[0], 4),
                ),
                "attention_mask": attention_mask,
            }
