import functools
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numba import njit
from torch.utils.data import DataLoader, Dataset

VOCAB_SIZE = 14

SPECIAL_TOKENS = ["<start>", "<next_line>", "<input_output_separator>", "<end>"]
TOKEN_TO_ID: Dict[str, int] = {str(i): i for i in range(10)}
for offset, token in enumerate(SPECIAL_TOKENS, start=10):
    TOKEN_TO_ID[token] = offset

START_TOKEN_ID = TOKEN_TO_ID["<start>"]
NEXT_LINE_TOKEN_ID = TOKEN_TO_ID["<next_line>"]
IO_SEPARATOR_TOKEN_ID = TOKEN_TO_ID["<input_output_separator>"]
END_TOKEN_ID = TOKEN_TO_ID["<end>"]

MAX_SEQ_LEN = 1863
IGNORE_INDEX = -100


@dataclass
class SequenceExample:
    tokens: torch.LongTensor
    example_id: int
    task_id: str
    split: str
    pair_index: int
    has_output: bool
    seq_len: int


class ColorAugmentor:
    def __init__(
        self,
        mappings: Sequence[torch.Tensor],
        apply_to_test_split: bool = False,
        seed: int = 42,
    ) -> None:
        self.mappings = list(mappings)
        self.apply_to_test_split = apply_to_test_split
        self.seed = seed
        self._epoch = 0
        self._cached_index = 0
        self._compute_index()

    @property
    def num_permutations(self) -> int:
        return len(self.mappings)

    @property
    def current_index(self) -> int:
        return self._cached_index

    def set_index(self, index: int) -> None:
        if self.num_permutations == 0:
            return
        self._epoch = max(0, int(index))
        self._compute_index()

    def _compute_index(self) -> None:
        N = self.num_permutations
        if N == 0:
            self._cached_index = 0
            return

        cycle = self._epoch // N
        step = self._epoch % N

        if step == 0 or N <= 1:
            self._cached_index = 0
            return

        g = torch.Generator()
        g.manual_seed(self.seed + cycle)

        perm = torch.randperm(N - 1, generator=g)
        random_offset = perm[step - 1].item()
        self._cached_index = random_offset + 1

    def mapping_for_split(self, split: str) -> Optional[torch.Tensor]:
        if not self.mappings:
            return None
        if split == "test" and not self.apply_to_test_split:
            return None
        return self.mappings[self.current_index]


class ARCExampleDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        splits: Sequence[str] = ("train", "test"),
        include_outputs: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
        drop_long_sequences: bool = False,
        task_whitelist: Optional[Sequence[str]] = None,
        load_test_solutions: bool = False,
    ) -> None:
        available_splits = {"train", "test"}
        for split in splits:
            if split not in available_splits:
                raise ValueError(
                    f"Unsupported split '{split}'. Expected values in {available_splits}."
                )

        self.source_path = Path(json_path)
        self.max_seq_len = max_seq_len
        self.drop_long_sequences = drop_long_sequences
        self.include_outputs = include_outputs

        challenges = load_challenges(self.source_path)

        solutions_map = {}
        if load_test_solutions:
            sol_path = self.source_path.with_name("solutions.json")
            if sol_path.exists():
                with sol_path.open("r") as handle:
                    solutions_map = json.load(handle)
            else:
                print(f"Warning: solutions.json not found at {sol_path}")

        if task_whitelist is not None:
            task_ids = list(task_whitelist)
            missing = [task_id for task_id in task_ids if task_id not in challenges]
            if missing:
                raise ValueError(f"Task ids {missing} were not found in {json_path}.")
        else:
            task_ids = sorted(challenges.keys())

        self.examples: List[SequenceExample] = []
        self.task_id_to_example_id: Dict[str, int] = {}
        self.indices_by_split: Dict[str, List[int]] = {split: [] for split in splits}
        self.task_ids = task_ids
        self.sequence_lengths: List[int] = []

        for example_id, task_id in enumerate(task_ids):
            self.task_id_to_example_id[task_id] = example_id
            task = challenges[task_id]
            for split in splits:
                pairs = task.get(split, [])
                for pair_index, pair in enumerate(pairs):
                    input_grid = pair["input"]
                    output_grid = pair.get("output")

                    if split == "test" and load_test_solutions:
                        if task_id in solutions_map:
                            task_sols = solutions_map[task_id]
                            if pair_index < len(task_sols):
                                output_grid = task_sols[pair_index]

                    has_output = output_grid is not None
                    include_output_tokens = include_outputs and has_output
                    append_end = include_output_tokens

                    tokens = encode_example(
                        input_grid,
                        output_grid,
                        include_output=include_output_tokens,
                        append_end=append_end,
                    )

                    if len(tokens) > max_seq_len:
                        if drop_long_sequences:
                            continue
                        raise ValueError(
                            f"Sequence length {len(tokens)} exceeds max_seq_len={max_seq_len} "
                            f"for task {task_id} ({split} pair {pair_index})."
                        )

                    tensor = torch.tensor(tokens, dtype=torch.long)
                    seq_len = len(tokens)
                    example = SequenceExample(
                        tokens=tensor,
                        example_id=example_id,
                        task_id=task_id,
                        split=split,
                        pair_index=pair_index,
                        has_output=has_output,
                        seq_len=seq_len,
                    )

                    self.indices_by_split.setdefault(split, []).append(
                        len(self.examples)
                    )
                    self.examples.append(example)
                    self.sequence_lengths.append(seq_len)

        self.num_examples = len(self.task_id_to_example_id)

        for ex in self.examples:
            fake_batch = ex.tokens.unsqueeze(0)
            mask = torch.ones_like(fake_batch, dtype=torch.bool)
            pos = compute_positions_3d(fake_batch, mask)
            ex.cached_positions = pos.squeeze(0)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SequenceExample:
        return self.examples[idx]

    def get_task_example_id(self, task_id: str) -> int:
        return self.task_id_to_example_id[task_id]


def generate_color_permutations(
    max_permutations: int, seed: int
) -> List[Tuple[int, ...]]:
    if max_permutations <= 0:
        return []
    rng = random.Random(seed)
    digits = list(range(1, 10))
    identity = tuple(digits)
    permutations: List[Tuple[int, ...]] = [identity]
    seen = {identity}
    limit = math.factorial(9)
    target = min(max_permutations, limit)
    if target == 1:
        return permutations

    if target == limit:
        all_perms = list(itertools.permutations(digits))
        rng.shuffle(all_perms)
        deduped = [identity]
        for perm in all_perms:
            if perm == identity:
                continue
            deduped.append(perm)
        return deduped[:target]

    while len(permutations) < target:
        perm = tuple(rng.sample(digits, len(digits)))
        if perm in seen:
            continue
        seen.add(perm)
        permutations.append(perm)
    return permutations


def color_permutation_to_mapping(perm: Sequence[int]) -> torch.Tensor:
    mapping = torch.arange(VOCAB_SIZE, dtype=torch.long)
    mapping[1:10] = torch.tensor(list(perm), dtype=torch.long)
    return mapping


def generate_color_mapping_tensors(
    max_permutations: int, seed: int
) -> List[torch.Tensor]:
    perms = generate_color_permutations(max_permutations, seed)
    return [color_permutation_to_mapping(perm) for perm in perms]


def _value_to_token_id(value: int) -> int:
    if value not in range(10):
        raise ValueError(f"Grid values must be digits in [0, 9], received {value}")
    return value


def grid_to_tokens(grid: Iterable[Iterable[int]]) -> List[int]:
    tokens: List[int] = []
    for row in grid:
        for value in row:
            tokens.append(_value_to_token_id(int(value)))
        tokens.append(NEXT_LINE_TOKEN_ID)
    return tokens


def encode_example(
    input_grid: Iterable[Iterable[int]],
    output_grid: Optional[Iterable[Iterable[int]]] = None,
    include_output: bool = True,
    append_end: bool = True,
) -> List[int]:
    tokens = [START_TOKEN_ID]
    tokens.extend(grid_to_tokens(input_grid))
    tokens.append(IO_SEPARATOR_TOKEN_ID)
    if include_output and output_grid is not None:
        tokens.extend(grid_to_tokens(output_grid))
    if append_end:
        tokens.append(END_TOKEN_ID)
    return tokens


def load_challenges(json_path: Path) -> Dict[str, dict]:
    with Path(json_path).open("r") as handle:
        return json.load(handle)


@njit
def _fill_3d_positions_numba(ids, mask, out, start_id, sep_id, end_id, nl_id):
    B, S = ids.shape
    for b in range(B):
        x = 0
        y = 0
        z = 1
        for t in range(S):
            if not mask[b, t]:
                continue

            val = ids[b, t]

            if val == start_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 0
                x = 0
                y = 0
                z = 1
                continue

            if val == sep_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 2
                x = 0
                y = 0
                z = 3
                continue

            if val == end_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 4
                continue

            px = x
            if px < 0:
                px = 0
            if px > 30:
                px = 30

            py = y
            if py < 0:
                py = 0
            if py > 29:
                py = 29

            out[b, t, 0] = px
            out[b, t, 1] = py
            out[b, t, 2] = z

            if val == nl_id:
                x = 0
                y += 1
            else:
                x += 1


def compute_positions_3d(
    input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq_len].")

    ids_cpu = input_ids.detach().cpu()
    ids_np = ids_cpu.numpy()

    if attention_mask is None:
        mask_np = np.ones_like(ids_np, dtype=bool)
    else:
        mask_np = attention_mask.detach().cpu().numpy().astype(bool)

    B, S = ids_np.shape
    pos_np = np.zeros((B, S, 3), dtype=np.int64)

    _fill_3d_positions_numba(
        ids_np,
        mask_np,
        pos_np,
        START_TOKEN_ID,
        IO_SEPARATOR_TOKEN_ID,
        END_TOKEN_ID,
        NEXT_LINE_TOKEN_ID,
    )

    return torch.from_numpy(pos_np).to(device=input_ids.device)


def collate_examples(
    batch: List[SequenceExample],
    pad_token_id: int = END_TOKEN_ID,
    color_mapper: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch encountered during collation.")

    batch_size = len(batch)
    max_len = MAX_SEQ_LEN

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    example_ids = torch.zeros(batch_size, dtype=torch.long)
    positions_3d = torch.zeros((batch_size, max_len, 3), dtype=torch.long)

    for idx, example in enumerate(batch):
        seq_len = example.seq_len
        tokens = example.tokens
        if color_mapper is not None:
            mapping = color_mapper(example.split)
            if mapping is not None:
                tokens = mapping[tokens]
        input_ids[idx, :seq_len] = tokens
        attention_mask[idx, :seq_len] = True
        example_ids[idx] = example.example_id
        positions_3d[idx, :seq_len] = example.cached_positions

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "example_ids": example_ids,
        "positions_3d": positions_3d,
        "task_ids": [example.task_id for example in batch],
        "splits": [example.split for example in batch],
        "has_output": [example.has_output for example in batch],
    }


def create_dataloader(
    dataset: ARCExampleDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    color_mapper: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
) -> DataLoader:
    collate_fn = (
        functools.partial(collate_examples, color_mapper=color_mapper)
        if color_mapper is not None
        else collate_examples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _build_color_augmentor(args: Any, is_eval: bool) -> Optional[ColorAugmentor]:
    flag_name = "enable_color_aug_eval" if is_eval else "enable_color_aug_train"
    max_name = "max_color_augments_eval" if is_eval else "max_color_augments_train"
    enabled = bool(getattr(args, flag_name, False))
    max_augments = int(getattr(args, max_name, 0) or 0)
    if not enabled or max_augments <= 0:
        return None

    seed = getattr(args, "color_aug_seed", None)
    if seed is None:
        seed = args.seed
    seed = int(seed)

    mappings = generate_color_mapping_tensors(max_augments, seed)
    if not mappings:
        return None

    return ColorAugmentor(
        mappings=mappings,
        apply_to_test_split=True if is_eval else False,
        seed=seed,
    )


def augment_color(
    batch: Dict[str, Any],
    *,
    color_augmentor: ColorAugmentor,
) -> None:
    splits = batch.get("splits")
    if not splits:
        return

    input_ids: torch.Tensor = batch["input_ids"]
    aug_map = color_augmentor.mappings[color_augmentor.current_index]
    vocab_size = aug_map.numel()

    should_aug = torch.tensor(
        [(color_augmentor.mapping_for_split(s) is not None) for s in splits],
        dtype=torch.bool,
    ).reshape(-1, 1)

    if not bool(should_aug.any()):
        return

    identity = torch.arange(vocab_size, dtype=torch.long)
    batch_maps = torch.where(should_aug, aug_map, identity)
    batch["input_ids"] = torch.gather(batch_maps, 1, input_ids)


def build_torch_data(
    args: Any,
) -> Tuple[
    ARCExampleDataset,
    DataLoader,
    Optional[DataLoader],
    Path,
    Optional[ColorAugmentor],
]:
    data_path = Path(args.data_path)

    train_dataset = ARCExampleDataset(
        json_path=data_path,
        splits=("train", "test"),
        include_outputs=True,
        max_seq_len=MAX_SEQ_LEN,
    )

    color_augmentor = _build_color_augmentor(args, is_eval=False)
    collate_color_mapper = (
        color_augmentor.mapping_for_split
        if color_augmentor is not None and getattr(args, "num_workers", 0) == 0
        else None
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        color_mapper=collate_color_mapper,
    )

    if color_augmentor is not None:
        train_loader.color_augmentor = color_augmentor
        train_loader.color_aug_in_collate = collate_color_mapper is not None

    val_loader = None
    if getattr(args, "do_validate", True):
        val_dataset = ARCExampleDataset(
            json_path=data_path,
            splits=("test",),
            include_outputs=True,
            load_test_solutions=True,
            max_seq_len=MAX_SEQ_LEN,
            task_whitelist=train_dataset.task_ids,
        )
        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=getattr(args, "val_batch_size", args.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
        )

    return train_dataset, train_loader, val_loader, data_path, color_augmentor
