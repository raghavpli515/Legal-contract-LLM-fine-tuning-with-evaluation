"""Contract-level train/val/test split, stratified on clause-category presence.

Every example derived from a contract lands in that contract's split, so no contract
text is shared across splits (tested). Stratification uses iterative stratification
(Sechidis et al., 2011) over the set of clause categories each contract contains:
the rarest categories are placed first, so rare clauses such as "Source Code Escrow"
(13 of 510 contracts) still appear in val and test.
"""

from __future__ import annotations

import random
from collections.abc import Mapping


def iterative_split(
    labels: Mapping[str, set[str]], fractions: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Assign each item (contract id -> its label set) to a split name."""
    rng = random.Random(seed)
    splits = list(fractions)
    order = sorted(labels)
    rng.shuffle(order)

    capacity = {s: fractions[s] * len(order) for s in splits}
    label_counts: dict[str, int] = {}
    for item_labels in labels.values():
        for label in item_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    wanted = {lab: {s: fractions[s] * n for s in splits} for lab, n in label_counts.items()}

    assignment: dict[str, str] = {}

    def assign(item: str, split: str) -> None:
        assignment[item] = split
        capacity[split] -= 1
        for label in labels[item]:
            wanted[label][split] -= 1

    while True:
        remaining = [i for i in order if i not in assignment]
        live: dict[str, list[str]] = {}
        for item in remaining:
            for label in labels[item]:
                live.setdefault(label, []).append(item)
        if not live:
            break
        rarest = min(live, key=lambda lab: (len(live[lab]), lab))
        for item in live[rarest]:
            best = max(splits, key=lambda s: (wanted[rarest][s], capacity[s], -splits.index(s)))
            assign(item, best)

    for item in order:  # items with no labels
        if item not in assignment:
            assign(item, max(splits, key=lambda s: (capacity[s], -splits.index(s))))
    return assignment
