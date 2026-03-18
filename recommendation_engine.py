from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from data_manager import Player
from team_balancer import TeamResult, weighted_score


@dataclass(frozen=True)
class Recommendation:
    swap_a: Player
    swap_b: Player
    before_diff: float
    after_diff: float
    improvement_points: float
    improvement_pct: float
    reason: str
    metric: str = "weighted"


def _key_pair(a: str, b: str) -> str:
    return "||".join(sorted([a, b]))


def recommend_swaps(
    result: TeamResult,
    config: Dict[str, Any],
    learning: Dict[str, Any] | None = None,
    top_k: int = 5,
) -> List[Recommendation]:
    """
    Recomienda swaps 1x1 entre equipos que mejoren el balance.
    Integra aprendizaje simple:
    - penaliza swaps muy repetidos (si históricamente se “deshacen”)
    - prioriza jugadores con alta frecuencia de movimiento (más “inestables”)
    """
    learning = learning or {"swap_counts": {}, "move_counts": {}}
    swap_counts: Dict[str, int] = learning.get("swap_counts", {}) or {}
    move_counts: Dict[str, int] = learning.get("move_counts", {}) or {}

    base_diff = float(result.diff)
    recs: List[Tuple[float, Recommendation]] = []

    a = list(result.team_a)
    b = list(result.team_b)
    weights = config.get("weights", {"physical": 0.4, "football": 0.6})
    for pa in a:
        for pb in b:
            new_a = [p for p in a if p.name != pa.name] + [pb]
            new_b = [p for p in b if p.name != pb.name] + [pa]

            # Evaluación ligera: diff ponderado del swap
            delta_w = weighted_score(pb, weights) - weighted_score(pa, weights)
            after = abs((result.score_a + delta_w) - (result.score_b - delta_w))
            if after >= base_diff:
                continue

            imp_pts = base_diff - after
            imp_pct = (imp_pts / base_diff * 100.0) if base_diff > 1e-9 else 0.0

            # Learning adjustments
            pair_key = _key_pair(pa.name, pb.name)
            repeated_penalty = 1.0 + 0.15 * float(swap_counts.get(pair_key, 0))
            move_bonus = 1.0 + 0.03 * float(move_counts.get(pa.name, 0) + move_counts.get(pb.name, 0))

            # score alto = mejor (beneficio / penalización) * bonus
            score = (imp_pts / repeated_penalty) * move_bonus

            reason = (
                f"Mejora el balance de {base_diff:.2f} → {after:.2f} "
                f"({imp_pct:.1f}% mejor). "
                f"Swap propuesto porque compensa la diferencia de puntuación ponderada entre equipos."
            )
            recs.append(
                (score, Recommendation(pa, pb, base_diff, after, imp_pts, imp_pct, reason, metric="weighted"))
            )

    recs.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in recs[:top_k]]


def recommend_swaps_by_metric(
    result: TeamResult,
    metric: str,
    top_k: int = 5,
) -> List[Recommendation]:
    """
    Recomendaciones SIN re-balancear, optimizando:
    - metric='football': minimizar |sum_football_a - sum_football_b|
    - metric='physical': minimizar |sum_physical_a - sum_physical_b|
    """
    a = list(result.team_a)
    b = list(result.team_b)

    if metric == "football":
        base_sum_a = float(result.football_a)
        base_sum_b = float(result.football_b)
        label = "fútbol"

        def score_of(p: Player) -> float:
            return float(p.football)

    else:
        base_sum_a = float(result.physical_a)
        base_sum_b = float(result.physical_b)
        label = "físico"

        def score_of(p: Player) -> float:
            return float(p.physical)

    base = abs(base_sum_a - base_sum_b)

    recs: List[Tuple[float, Recommendation]] = []
    for pa in a:
        for pb in b:
            # swap effect on raw sums
            delta = score_of(pb) - score_of(pa)
            after = abs((base_sum_a + delta) - (base_sum_b - delta))
            if after >= base:
                continue
            imp = base - after
            imp_pct = (imp / base * 100.0) if base > 1e-9 else 0.0
            reason = f"Equilibra {label}: {base:.2f} → {after:.2f} ({imp_pct:.1f}% mejor)."
            recs.append((imp, Recommendation(pa, pb, base, after, imp, imp_pct, reason, metric=metric)))

    recs.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in recs[:top_k]]


def make_group_id() -> str:
    return str(int(time.time() * 1000))

