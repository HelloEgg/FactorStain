from __future__ import annotations

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


@dataclass(frozen=True)
class RenderingEpisode:
    source: int
    target: int
    pair_type: str


def enumerate_factorial_episodes(index: pd.DataFrame) -> list[FactorialEpisode]:
    """Enumerate complete 2×2 acquisition rectangles within aligned groups."""
    episodes: list[FactorialEpisode] = []
    for _, group in index.groupby("aligned_group_id"):
        lookup = {
            (str(row.stain_id), str(row.scanner_id)): int(i)
            for i, row in group.iterrows()
        }
        stains = sorted({key[0] for key in lookup})
        scanners = sorted({key[1] for key in lookup})
        for ia, stain_a in enumerate(stains):
            for stain_b in stains[ia + 1 :]:
                for ix, scanner_x in enumerate(scanners):
                    for scanner_y in scanners[ix + 1 :]:
                        keys = [
                            (stain_a, scanner_x),
                            (stain_a, scanner_y),
                            (stain_b, scanner_x),
                            (stain_b, scanner_y),
                        ]
                        if all(key in lookup for key in keys):
                            episodes.append(
                                FactorialEpisode(*(lookup[key] for key in keys))
                            )
    return episodes


def enumerate_rendering_episodes(
    index: pd.DataFrame,
    source_cells: set[tuple[str, str]],
    target_cells: set[tuple[str, str]] | None = None,
    morphology_split: str | None = None,
    max_per_group: int = 64,
    seed: int = 42,
) -> list[RenderingEpisode]:
    """Build balanced scanner-paired, cross-stain, and factorial episodes.

    Scanner-paired episodes share aligned_group_id and stain_id and are safe for
    strong pixel supervision. Every episode involving a stain change is labeled
    separately so serial sections never receive blind pixel-perfect loss.
    """
    target_cells = source_cells if target_cells is None else target_cells
    frame = index
    if morphology_split is not None:
        frame = frame[frame.morphology_split.eq(morphology_split)]
    if "image_exists" in frame:
        frame = frame[frame.image_exists]
    frame = frame.copy()
    frame["cell"] = list(zip(frame.stain_id.astype(str), frame.scanner_id.astype(str)))
    rng = np.random.default_rng(seed)
    episodes: list[RenderingEpisode] = []
    for _, group in frame.groupby("aligned_group_id", sort=True):
        sources = group[group.cell.isin(source_cells)]
        targets = group[group.cell.isin(target_cells)]
        candidates: dict[str, list[RenderingEpisode]] = {
            "scanner_pair": [],
            "cross_stain": [],
            "factorial": [],
        }
        for target_index, target in targets.iterrows():
            for source_index, source in sources.iterrows():
                if source_index == target_index:
                    continue
                same_stain = str(source.stain_id) == str(target.stain_id)
                same_scanner = str(source.scanner_id) == str(target.scanner_id)
                if same_stain and not same_scanner:
                    pair_type = "scanner_pair"
                elif not same_stain and same_scanner:
                    pair_type = "cross_stain"
                elif not same_stain and not same_scanner:
                    pair_type = "factorial"
                else:
                    continue
                candidates[pair_type].append(
                    RenderingEpisode(int(source_index), int(target_index), pair_type)
                )
        group_episodes: list[RenderingEpisode] = []
        nonempty = [name for name, values in candidates.items() if values]
        if nonempty:
            quota = max(1, max_per_group // len(nonempty))
            for name in nonempty:
                values = candidates[name]
                if len(values) > quota:
                    selected = rng.choice(len(values), quota, replace=False)
                    values = [values[int(position)] for position in selected]
                group_episodes.extend(values)
            if len(group_episodes) > max_per_group:
                selected = rng.choice(len(group_episodes), max_per_group, replace=False)
                group_episodes = [
                    group_episodes[int(position)] for position in selected
                ]
        episodes.extend(group_episodes)
    return episodes


class FactorialEpisodeSampler(Sampler[list[int]]):
    """Yield dynamically rotated [A/X, A/Y, B/X, masked B/Y] episodes."""

    def __init__(
        self, episodes: list[FactorialEpisode], seed: int = 42, shuffle: bool = True
    ) -> None:
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
