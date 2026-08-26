import pandas as pd

from factorstain.data.splits import build_combination_split, group_train_val_test_split


def factorial_frame(groups=20, stains=5, scanners=4):
    return pd.DataFrame(
        [
            {"aligned_group_id": f"g{group}", "stain_id": f"s{stain}", "scanner_id": f"q{scanner}"}
            for group in range(groups)
            for stain in range(stains)
            for scanner in range(scanners)
        ]
    )


def test_combination_split_retains_every_factor():
    frame = factorial_frame()
    split = build_combination_split(frame, 0.20, 42)
    train = {(x["stain_id"], x["scanner_id"]) for x in split["train_cells"]}
    heldout = {(x["stain_id"], x["scanner_id"]) for x in split["heldout_cells"]}
    assert train.isdisjoint(heldout)
    assert {s for s, _ in train} == set(split["stains"])
    assert {q for _, q in train} == set(split["scanners"])


def test_group_split_has_no_leakage():
    frame = factorial_frame()
    labels = group_train_val_test_split(frame, seed=42)
    combined = frame.assign(split=labels)
    assert combined.groupby("aligned_group_id").split.nunique().max() == 1
    assert set(labels) == {"train", "val", "test"}

