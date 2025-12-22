# src/lapf_project/data/text_dataset.py

import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor


def water_level_to_class(water_level: int, num_labels: int) -> int:
    if water_level == 100:
        return num_labels - 1
    return water_level // (100 // num_labels)


def data_to_pairs_for_LAPF(
    data: List[Dict],
    num_labels: int,
) -> List[Tuple[int, str]]:
    pair_set = set()  # type: set[Tuple[int, str]]

    for d in data:
        water_level = int(d["water_level"])
        text = d["text"]

        cls = water_level_to_class(water_level, num_labels)
        pair_set.add((cls, text))

    return list(pair_set)


def data_to_pairs_for_EDAPF(
    data: List[Dict],
) -> List[Tuple[int, str]]:
    pair_set = set()  # type: set[Tuple[int, str]]

    for d in data:
        water_level = int(d["water_level"])
        text = d["text"]

        pair_set.add((water_level, text))

    return list(pair_set)


def pairs_to_batches(
    pairs: Sequence[Tuple[int, str]],
    batch_size: int,
    device: torch.device | str,
    shuffle: bool = True,
    seed: int | None = None,
) -> List[Tuple[Tensor, List[str]]]:
    """Convert (class_id, text) pairs into mini-batches.

    Parameters
    ----------
    pairs : sequence of (class_id, text)
        Output of `data_to_pairs`, for example.
    batch_size : int
        Batch size.
    device : torch.device or str
        Device for the label tensor.
    shuffle : bool, default True
        Whether to shuffle the data before batching.
    seed : int or None, default None
        Random seed for shuffling. If None, use the global RNG state.

    Returns
    -------
    batches : list of (labels or levels, texts)
        labels : Tensor, shape (B,), on `device`
        texts  : list of str, length B
    """
    device = torch.device(device)

    indices = list(range(len(pairs)))
    if shuffle:
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(indices)

    batches: List[Tuple[Tensor, List[str]]] = []

    for start in range(0, len(indices), batch_size):
        idx_slice = indices[start : start + batch_size]
        samples = [pairs[i] for i in idx_slice]

        batch = (
            torch.tensor([s[0] for s in samples], device=device),
            [s[1] for s in samples],
        )

        batches.append(batch)

    return batches
