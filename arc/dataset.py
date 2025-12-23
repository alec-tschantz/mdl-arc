import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

MAX_SIZE = 30
NUM_COLORS = 10

PAD_TOKEN_ID = NUM_COLORS
MASK_TOKEN_ID = NUM_COLORS + 1
VOCAB_SIZE = NUM_COLORS + 2

GRID_LEN = MAX_SIZE * MAX_SIZE


def _pad_grid(grid):
    arr = np.full((MAX_SIZE, MAX_SIZE), PAD_TOKEN_ID, dtype=np.int32)
    mask = np.zeros((MAX_SIZE, MAX_SIZE), dtype=bool)
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return arr, mask
    arr[:h, :w] = np.array(grid, dtype=np.int32)
    mask[:h, :w] = True
    return arr, mask


class Dataset:
    def __init__(
        self,
        path: Path,
        split: str,
        subset: str = "train",
        task_lookup: Optional[Dict[str, int]] = None,
        *,
        extra_train_path: Optional[Path] = None,
        seed: int = 0,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> None:
        self.path = Path(path)
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.task_lookup: Dict[str, int] = (
            dict(task_lookup) if task_lookup is not None else {}
        )

        inputs_list = []
        targets_list = []
        input_masks = []
        task_id_list = []

        def _process_dir(directory: Path):
            files = sorted(directory.glob("*.json"))
            examples_key = "train" if self.subset == "train" else "test"

            for file_path in files:
                task_name = file_path.stem
                if self.subset == "train":
                    if task_name not in self.task_lookup:
                        self.task_lookup[task_name] = len(self.task_lookup)
                    task_index = self.task_lookup[task_name]
                else:
                    if task_name not in self.task_lookup:
                        continue
                    task_index = self.task_lookup[task_name]

                with file_path.open("r") as fh:
                    task_data = json.load(fh)

                examples = task_data.get(examples_key, [])
                for example in examples:
                    inp = example["input"]
                    out = example.get("output")
                    if out is None:
                        continue

                    h_in = len(inp)
                    w_in = len(inp[0]) if h_in else 0
                    h_out = len(out)
                    w_out = len(out[0]) if h_out else 0

                    max_cur_y = max(h_in, h_out)
                    max_cur_x = max(w_in, w_out)
                    if max_cur_y > MAX_SIZE or max_cur_x > MAX_SIZE:
                        continue

                    padded_inp, mask_inp = _pad_grid(inp)
                    padded_out, _ = _pad_grid(out)

                    inputs_list.append(padded_inp)
                    targets_list.append(padded_out)
                    input_masks.append(mask_inp)
                    task_id_list.append(task_index)

        _process_dir(self.path / split)
        if self.subset == "train" and extra_train_path is not None:
            _process_dir(Path(extra_train_path) / split)

        self.inputs = np.stack(inputs_list).astype(np.int32)
        self.targets = np.stack(targets_list).astype(np.int32)
        self.input_masks = np.stack(input_masks).astype(bool)
        self.task_ids = np.array(task_id_list, dtype=np.int32)
        self.num_samples = len(self.inputs)
        self.num_tasks = len(self.task_lookup)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        usable = self.num_samples - (self.num_samples % self.batch_size)

        for start in range(0, usable, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_size = batch_idx.shape[0]

            input_mask = self.input_masks[batch_idx].reshape(batch_size, -1)
            output_mask = np.ones((batch_size, GRID_LEN), dtype=bool)
            attention_mask = np.concatenate([input_mask, output_mask], axis=1)

            yield {
                "inputs": self.inputs[batch_idx],
                "targets": self.targets[batch_idx],
                "task_ids": self.task_ids[batch_idx],
                "attention_mask": attention_mask,
            }
