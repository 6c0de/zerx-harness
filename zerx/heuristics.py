"""No-GPU fallback/candidate layer: numpy-based click-candidate scoring and
a graded, decaying negative-affordance tracker (STRATEGY.md's "soft
negative affordances" — down-rank an ineffective object signature instead
of permanently banning it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from zerx.perception import LabeledObject, PerceptionResult


def _signature(obj: LabeledObject) -> Tuple[int, int]:
    """A coarse (color, size-bucket) signature that identifies "the same
    kind of thing" across frames, not the exact same object instance.
    """
    size_bucket = min(obj.size, 20) // 4
    return (obj.color, size_bucket)


class DeadSignatureTracker:
    """Per-signature penalty in [0.0, 1.0]. An ineffective click raises it
    by `penalty_step` (clamped at 1.0); an effective use of the same
    signature lowers it by `recovery_step` (clamped at 0.0). A signature
    that has never been observed has penalty 0.0 — it starts trusted.
    """

    def __init__(self, penalty_step: float = 0.35, recovery_step: float = 0.5) -> None:
        self._penalty: Dict[Tuple[int, int], float] = {}
        self._penalty_step = penalty_step
        self._recovery_step = recovery_step

    def record_outcome(self, obj: LabeledObject, effective: bool) -> None:
        sig = _signature(obj)
        current = self._penalty.get(sig, 0.0)
        if effective:
            current = max(0.0, current - self._recovery_step)
        else:
            current = min(1.0, current + self._penalty_step)
        if current == 0.0:
            self._penalty.pop(sig, None)
        else:
            self._penalty[sig] = current

    def penalty(self, obj: LabeledObject) -> float:
        return self._penalty.get(_signature(obj), 0.0)

    def reset(self) -> None:
        self._penalty.clear()


@dataclass(frozen=True)
class ClickCandidate:
    x: int
    y: int
    object_label: str
    score: float


def _object_center(obj: LabeledObject) -> Tuple[int, int]:
    min_x, min_y, max_x, max_y = obj.bbox
    return ((min_x + max_x) // 2, (min_y + max_y) // 2)


def rank_click_candidates(
    perception: PerceptionResult,
    affordance: DeadSignatureTracker,
    grid_size: int = 64,
) -> List[ClickCandidate]:
    """Score objects by "small, rare-colored, button-like" heuristics, then
    scale each score by `(1 - penalty)`. Smaller objects and less-common
    colors score higher; penalized signatures rank lower but are always
    still returned — ranking, not exclusion, is how "soft" affordances
    stay soft.
    """
    objects = perception.objects
    if not objects:
        return []

    sizes = np.array([o.size for o in objects], dtype=np.float64)
    colors = [o.color for o in objects]
    color_counts = {c: colors.count(c) for c in set(colors)}
    rarity = np.array([1.0 / color_counts[c] for c in colors], dtype=np.float64)

    max_size = sizes.max()
    size_score = 1.0 - (sizes / max_size) if max_size > 0 else np.zeros_like(sizes)
    max_rarity = rarity.max()
    rarity_score = rarity / max_rarity if max_rarity > 0 else np.zeros_like(rarity)

    combined = 0.5 * size_score + 0.5 * rarity_score

    candidates = []
    for obj, base_score in zip(objects, combined):
        score = float(base_score) * (1.0 - affordance.penalty(obj))
        cx, cy = _object_center(obj)
        cx = min(max(cx, 0), grid_size - 1)
        cy = min(max(cy, 0), grid_size - 1)
        candidates.append(ClickCandidate(x=cx, y=cy, object_label=obj.label, score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def size_rarity_scores(sizes: Tuple[int, ...], colors: Tuple[int, ...]) -> List[float]:
    """Pure "small objects and rare colors score higher" formula --
    factored out of rank_click_candidates so zerx.scene.list_salient_objects
    can reuse the same scoring core instead of duplicating it. Unlike
    rank_click_candidates, this takes no affordance tracker and applies no
    penalty; it exists so both callers share one formula.
    """
    if not sizes:
        return []
    sizes_arr = np.array(sizes, dtype=np.float64)
    color_counts: Dict[int, int] = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    rarity = np.array([1.0 / color_counts[c] for c in colors], dtype=np.float64)

    max_size = sizes_arr.max()
    size_score = 1.0 - (sizes_arr / max_size) if max_size > 0 else np.zeros_like(sizes_arr)
    max_rarity = rarity.max()
    rarity_score = rarity / max_rarity if max_rarity > 0 else np.zeros_like(rarity)

    combined = 0.5 * size_score + 0.5 * rarity_score
    return [float(v) for v in combined]
