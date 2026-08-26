from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


@dataclass(frozen=True)
class FactorialEpisode:
    ax: int
    ay: int
    bx: int
    by: int


def enumerate_factorial_episodes(index: pd.DataFrame) -> list[FactorialEpisode]:
    """Enumerate complete 2×2 acquisition rectangles within aligned groups."""
    episodes: list[FactorialEpisode] = []
    for _, group in index.groupby("aligned_group_id"):
        lookup = {(str(row.stain_id), str(row.scanner_id)): int(i) for i, row in group.iterrows()}
        stains = sorted({key[0] for key in lookup})
        scanners = sorted({key[1] for key in lookup})
        for ia, stain_a in enumerate(stains):
            for stain_b in stains[ia + 1 :]:
                for ix, scanner_x in enumerate(scanners):
                    for scanner_y in scanners[ix + 1 :]:
                        keys = [(stain_a, scanner_x), (stain_a, scanner_y), (stain_b, scanner_x), (stain_b, scanner_y)]
                        if all(key in lookup for key in keys):
                            episodes.append(FactorialEpisode(*(lookup[key] for key in keys)))
    return episodes


class FactorialEpisodeSampler(Sampler[list[int]]):
    """Yield dynamically rotated [A/X, A/Y, B/X, masked B/Y] episodes."""
    def __init__(self, episodes: list[FactorialEpisode], seed: int = 42, shuffle: bool = True) -> None:
        self.episodes = episodes
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(len(self.episodes))
        if self.shuffle:
            rng.shuffle(order)
        for idx in order:
            episode = self.episodes[int(idx)]
            yield [episode.ax, episode.ay, episode.bx, episode.by]

    def __len__(self) -> int:
        return len(self.episodes)

