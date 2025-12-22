import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

MAX_SIZE = 30
NUM_COLORS = 10
MAX_COLOR_ID = NUM_COLORS - 1

IGNORE_TOKEN_ID = 10
IO_SEPARATOR_TOKEN_ID = 11
NEXT_LINE_TOKEN_ID = 12
END_TOKEN_ID = 13
START_TOKEN_ID = 14

VOCAB_SIZE = 15

GRID_SEQ_LEN = MAX_SIZE * (MAX_SIZE + 1)
SEQ_LEN = GRID_SEQ_LEN * 2 + 3


def _pad_grid(grid):
    arr = np.full((MAX_SIZE, MAX_SIZE), IGNORE_TOKEN_ID, dtype=np.int32)
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return arr
    arr[:h, :w] = np.array(grid, dtype=np.int32)
    return arr


def _encode_grid(grid):
    padded = _pad_grid(grid)
    tokens = []
    for row in padded:
        tokens.extend(row.tolist())
        tokens.append(NEXT_LINE_TOKEN_ID)
    return tokens


def encode_example(input_grid, output_grid):
    tokens = [START_TOKEN_ID]
    tokens.extend(_encode_grid(input_grid))
    tokens.append(IO_SEPARATOR_TOKEN_ID)
    tokens.extend(_encode_grid(output_grid))
    tokens.append(END_TOKEN_ID)
    tokens = np.array(tokens, dtype=np.int32)
    assert tokens.shape[0] == SEQ_LEN
    return tokens


def _build_positions(tokens):
    pos = np.zeros((len(tokens), 3), dtype=np.int32)
    x = 0
    y = 0
    z = 1
    for idx, tok in enumerate(tokens):
        if tok == START_TOKEN_ID:
            pos[idx] = (0, 0, 0)
            x = 0
            y = 0
            z = 1
            continue
        if tok == IO_SEPARATOR_TOKEN_ID:
            pos[idx] = (0, 0, 2)
            x = 0
            y = 0
            z = 3
            continue
        if tok == END_TOKEN_ID:
            pos[idx] = (0, 0, 4)
            continue

        px = x
        py = y
        if px < 0:
            px = 0
        if px > MAX_SIZE:
            px = MAX_SIZE
        if py < 0:
            py = 0
        if py > MAX_SIZE - 1:
            py = MAX_SIZE - 1

        pos[idx] = (px, py, z)

        if tok == NEXT_LINE_TOKEN_ID:
            x = 0
            y += 1
        else:
            x += 1
    return pos


_TEMPLATE_TOKENS = encode_example([[0]], [[0]])
POSITIONS_TEMPLATE = _build_positions(_TEMPLATE_TOKENS)
ATTENTION_MASK_TEMPLATE = np.ones((SEQ_LEN,), dtype=bool)


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

        tokens_list = []
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
                    w_in = len(inp[0])
                    h_out = len(out)
                    w_out = len(out[0])

                    max_cur_y = max(h_in, h_out)
                    max_cur_x = max(w_in, w_out)

                    if max_cur_y > MAX_SIZE or max_cur_x > MAX_SIZE:
                        continue

                    tokens_list.append(encode_example(inp, out))
                    task_id_list.append(task_index)

        _process_dir(self.path / split)
        if self.subset == "train" and extra_train_path is not None:
            _process_dir(Path(extra_train_path))

        self.tokens = np.stack(tokens_list).astype(np.int32)
        self.task_ids = np.array(task_id_list, dtype=np.int32)
        self.num_samples = len(self.tokens)
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

            yield {
                "tokens": self.tokens[batch_idx],
                "task_ids": self.task_ids[batch_idx],
                "attention_mask": np.broadcast_to(
                    ATTENTION_MASK_TEMPLATE, (batch_size, SEQ_LEN)
                ),
                "positions": np.broadcast_to(
                    POSITIONS_TEMPLATE, (batch_size, SEQ_LEN, 3)
                ),
            }
