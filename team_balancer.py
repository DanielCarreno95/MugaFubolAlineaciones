from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_manager import Player, _slug


@dataclass(frozen=True)
class TeamResult:
    team_a: Tuple[Player, ...]
    team_b: Tuple[Player, ...]
    score_a: float
    score_b: float
    diff: float
    pos_summary_a: Dict[str, int]
    pos_summary_b: Dict[str, int]
    note: str = ""
    football_a: float = 0.0
    football_b: float = 0.0
    physical_a: float = 0.0
    physical_b: float = 0.0


def weighted_score(p: Player, weights: Dict[str, float]) -> float:
    return float(weights.get("physical", 0.4)) * float(p.physical) + float(weights.get("football", 0.6)) * float(p.football)


def _map_role(pos: str, position_map: Dict[str, str]) -> str:
    key = _slug(pos)
    return position_map.get(key, position_map.get(key.replace("_", ""), "UNK"))


def _team_role_counts(team: Sequence[Player], position_map: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0, "UNK": 0}
    for p in team:
        r1 = _map_role(p.pos1, position_map)
        if r1 in counts:
            counts[r1] += 1
        else:
            counts["UNK"] += 1
    return counts


def _positional_penalty(
    team: Sequence[Player],
    rules: Dict[str, Dict[str, int]],
    position_map: Dict[str, str],
) -> Tuple[int, Dict[str, int]]:
    """
    Penaliza si el equipo NO puede cumplir la estructura usando pos1/pos2.
    Regla: priorizar pos1, usar pos2 para cubrir déficits.
    No penalizamos jugar fuera de posición => solo comprobamos posibilidad de cubrir mínimos razonables.
    """
    # Conteo por pos1
    counts1 = _team_role_counts(team, position_map)

    # Potencial extra por pos2 (solo si pos1 no es esa posición)
    extra: Dict[str, int] = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in team:
        r1 = _map_role(p.pos1, position_map)
        r2 = _map_role(p.pos2, position_map) if p.pos2 else "UNK"
        if r2 in extra and r2 != r1:
            extra[r2] += 1

    # Estimación: cuántos podríamos alcanzar por rol usando secundarios (sin sobre-optimizar)
    achievable = {k: counts1.get(k, 0) + extra.get(k, 0) for k in ("GK", "DEF", "MID", "FWD")}

    penalty = 0
    for role, lim in rules.items():
        mn = int(lim.get("min", 0))
        mx = int(lim.get("max", 999))
        got_min = achievable.get(role, 0)
        if got_min < mn:
            penalty += (mn - got_min) * 100  # fuerte: preferimos cumplir estructura
        # el máximo lo tratamos suave (no queremos forzar demasiado)
        got_primary = counts1.get(role, 0)
        if got_primary > mx:
            penalty += (got_primary - mx) * 5

    # Caso portero: si hay exactamente 1 GK global, luego lo asignamos fijo a un equipo (fuera de aquí)
    summary = {k: counts1.get(k, 0) for k in ("GK", "DEF", "MID", "FWD")}
    return penalty, summary


def _total_score(team: Sequence[Player], weights: Dict[str, float]) -> float:
    return sum(weighted_score(p, weights) for p in team)

def _topk_names(players: Sequence[Player], key: str, k: int) -> List[str]:
    if key == "football":
        arr = sorted(players, key=lambda p: float(p.football), reverse=True)
    else:
        arr = sorted(players, key=lambda p: float(p.physical), reverse=True)
    return [p.name for p in arr[:k]]


def _top_split_penalty(team_a: Sequence[Player], top_names_ordered: Sequence[str], weight: float) -> float:
    """
    Penaliza una distribución "apilada" de los Top-K.
    Regla: alternamos por ranking: índices pares => esperados en Equipo A, impares => esperados en B.
    Esto fuerza que el mejor y el 2º mejor no caigan juntos (y en general que no haya apilamiento por ranking).
    """
    set_a = set(p.name for p in team_a)
    mismatches = 0
    for idx, name in enumerate(top_names_ordered):
        expected_in_a = (idx % 2 == 0)
        actual_in_a = name in set_a
        if actual_in_a != expected_in_a:
            mismatches += 1
    # También forzamos 2 y 2 como mínimo de forma consistente (count soft constraint).
    target_in_a = len(top_names_ordered) // 2
    count_a = sum(1 for name in top_names_ordered if name in set_a)
    count_pen = weight * abs(int(count_a) - int(target_in_a))
    parity_pen = weight * float(mismatches)
    return count_pen + parity_pen


def _top_count_penalty(team_a: Sequence[Player], top_names: Sequence[str], target_in_a: int, weight: float) -> float:
    """Penaliza solo por el conteo de Top-N que caen en el Equipo A (sin fijar pares)."""
    set_a = set(p.name for p in team_a)
    count_a = sum(1 for name in top_names if name in set_a)
    return float(weight) * float(abs(int(count_a) - int(target_in_a)))


def balance_teams(
    players: Sequence[Player],
    config: Dict[str, Any],
    objective_mode: str = "weighted",
) -> TeamResult:
    """
    Balancea EXACTAMENTE 2*team_size jugadores en 2 equipos (team_size=7 u 8 típicamente).
    Estrategia: búsqueda por combinaciones (C(16,8)=12870 / C(14,7)=3432) con penalización posicional.
    """
    team_size = int(config.get("team_size", 8))
    if len(players) != 2 * team_size:
        raise ValueError(f"Se requieren exactamente {2*team_size} jugadores seleccionados (recibido: {len(players)}).")

    weights = config.get("weights", {"physical": 0.4, "football": 0.6})
    rules_by_size = config.get("positional_rules_by_team_size", {}) or {}
    rules = rules_by_size.get(str(team_size), config.get("positional_rules", {})) or {}
    position_map = config.get("position_map", {})

    # Reparto de tops (soft-constraint fuerte): Top 4 fútbol y Top 4 físico deben quedar 2 y 2.
    top_k = 4
    top_football = _topk_names(players, "football", top_k)
    top_physical = _topk_names(players, "physical", top_k)
    top_split_weight = float(config.get("top_split_weight", 75.0))  # configurable
    top2_football = top_football[:2]
    top2_physical = top_physical[:2]

    # Regla portero único
    gks = [p for p in players if _map_role(p.pos1, position_map) == "GK" or _map_role(p.pos2, position_map) == "GK"]
    fixed_gk: Optional[Player] = gks[0] if len(gks) == 1 else None

    best_obj: Optional[float] = None
    best: Optional[TeamResult] = None

    idx = list(range(len(players)))
    for comb in combinations(idx, team_size):
        team_a = [players[i] for i in comb]
        team_b = [players[i] for i in idx if i not in comb]

        sa = _total_score(team_a, weights)
        sb = _total_score(team_b, weights)
        diff = abs(sa - sb)

        fa = sum(p.football for p in team_a)
        fb = sum(p.football for p in team_b)
        pa = sum(p.physical for p in team_a)
        pb = sum(p.physical for p in team_b)

        pen_a, sum_a = _positional_penalty(team_a, rules, position_map)
        pen_b, sum_b = _positional_penalty(team_b, rules, position_map)
        if objective_mode == "sport":
            # objetivo: equilibrar fútbol y físico (independiente de ponderación)
            objective = abs(fa - fb) + abs(pa - pb) + pen_a + pen_b
        else:
            # objetivo ponderado pero también forzando equilibrio en fútbol y físico por separado
            # (evita casos donde se compensa fútbol vs físico y el total ponderado queda igual).
            c = 0.20  # intensidad extra del equilibrio fútbol/físico
            objective = diff + (
                c * (float(weights.get("football", 0.6)) * abs(fa - fb) + float(weights.get("physical", 0.4)) * abs(pa - pb))
            ) + pen_a + pen_b

        # Penalización Top split (2 y 2). Esto evita que los mejores se "apilen".
        objective += _top_split_penalty(team_a, top_football, top_split_weight)
        objective += _top_split_penalty(team_a, top_physical, top_split_weight)

        # Extra: los 2 mejores de cada métrica deben repartirse (1 y 1).
        # Esto evita casos como "los dos mejores de fútbol (o físico) siempre juntos".
        objective += _top_count_penalty(team_a, top2_football, target_in_a=1, weight=top_split_weight)
        objective += _top_count_penalty(team_a, top2_physical, target_in_a=1, weight=top_split_weight)

        # Portero único: asignarlo al equipo más débil (compensación implícita).
        # Si el GK cae en el equipo más fuerte, penalizamos (no lo prohibimos para no bloquear casos raros).
        if fixed_gk is not None:
            weaker_is_a = sa <= sb
            gk_in_a = fixed_gk in team_a
            if weaker_is_a != gk_in_a:
                objective += 50  # penalización moderada

        if best is None or best_obj is None or objective < best_obj:
            best_obj = objective
            best = TeamResult(
                team_a=tuple(team_a),
                team_b=tuple(team_b),
                score_a=sa,
                score_b=sb,
                diff=diff,  # diff real, sin penalización
                pos_summary_a=sum_a,
                pos_summary_b=sum_b,
                note=f"mode={objective_mode} | diff={diff:.2f}, pos_penalty={pen_a+pen_b}",
                football_a=fa,
                football_b=fb,
                physical_a=pa,
                physical_b=pb,
            )

    if best is None:
        raise RuntimeError("No se pudo generar una partición válida.")

    return best


def balance_2_options(
    players: Sequence[Player],
    config: Dict[str, Any],
    objective_mode: str = "weighted",
    min_team_a_diff: Optional[int] = None,
) -> Tuple[TeamResult, TeamResult]:
    """
    Devuelve 2 opciones lo más equilibradas posible, pero con equipos A diferentes.
    `min_team_a_diff` = mínimo de jugadores distintos entre el Equipo A de ambas opciones.
    """
    team_size = int(config.get("team_size", 8))
    if len(players) != 2 * team_size:
        raise ValueError(f"Se requieren exactamente {2*team_size} jugadores seleccionados (recibido: {len(players)}).")

    if min_team_a_diff is None:
        min_team_a_diff = 3 if team_size >= 8 else 2

    # 1) Mejor opción (mínimo objective)
    best1 = balance_teams(players, config, objective_mode=objective_mode)
    best1_set = frozenset(p.name for p in best1.team_a)

    # 2) Segunda mejor: misma lógica pero con diversidad en Equipo A
    weights = config.get("weights", {"physical": 0.4, "football": 0.6})
    rules_by_size = config.get("positional_rules_by_team_size", {}) or {}
    rules = rules_by_size.get(str(team_size), config.get("positional_rules", {})) or {}
    position_map = config.get("position_map", {})

    top_k = 4
    top_football = _topk_names(players, "football", top_k)
    top_physical = _topk_names(players, "physical", top_k)
    top_split_weight = float(config.get("top_split_weight", 75.0))
    top2_football = top_football[:2]
    top2_physical = top_physical[:2]

    gks = [p for p in players if _map_role(p.pos1, position_map) == "GK" or _map_role(p.pos2, position_map) == "GK"]
    fixed_gk: Optional[Player] = gks[0] if len(gks) == 1 else None

    best2_obj: Optional[float] = None
    best2: Optional[TeamResult] = None

    idx = list(range(len(players)))
    for comb in combinations(idx, team_size):
        team_a = [players[i] for i in comb]
        team_b = [players[i] for i in idx if i not in comb]

        sa = _total_score(team_a, weights)
        sb = _total_score(team_b, weights)
        diff = abs(sa - sb)

        fa = sum(p.football for p in team_a)
        fb = sum(p.football for p in team_b)
        pa = sum(p.physical for p in team_a)
        pb = sum(p.physical for p in team_b)

        pen_a, sum_a = _positional_penalty(team_a, rules, position_map)
        pen_b, sum_b = _positional_penalty(team_b, rules, position_map)

        if objective_mode == "sport":
            objective = abs(fa - fb) + abs(pa - pb) + pen_a + pen_b
        else:
            c = 0.20
            objective = diff + (
                c * (float(weights.get("football", 0.6)) * abs(fa - fb) + float(weights.get("physical", 0.4)) * abs(pa - pb))
            ) + pen_a + pen_b

        objective += _top_split_penalty(team_a, top_football, top_split_weight)
        objective += _top_split_penalty(team_a, top_physical, top_split_weight)

        objective += _top_count_penalty(team_a, top2_football, target_in_a=1, weight=top_split_weight)
        objective += _top_count_penalty(team_a, top2_physical, target_in_a=1, weight=top_split_weight)

        if fixed_gk is not None:
            weaker_is_a = sa <= sb
            gk_in_a = fixed_gk in team_a
            if weaker_is_a != gk_in_a:
                objective += 50

        team_a_set = frozenset(p.name for p in team_a)
        overlap = len(team_a_set & best1_set)
        diff_players = team_size - overlap
        if diff_players < min_team_a_diff:
            continue

        if best2 is None or best2_obj is None or objective < best2_obj:
            best2_obj = objective
            best2 = TeamResult(
                team_a=tuple(team_a),
                team_b=tuple(team_b),
                score_a=sa,
                score_b=sb,
                diff=diff,
                pos_summary_a=sum_a,
                pos_summary_b=sum_b,
                note=f"option=2 | diff={diff:.2f}",
                football_a=fa,
                football_b=fb,
                physical_a=pa,
                physical_b=pb,
            )

    # Si por alguna razón no se encuentra diversidad suficiente, caemos al siguiente mejor sin constraint duro.
    if best2 is None:
        best2 = balance_teams(players, config, objective_mode=objective_mode)

    return best1, best2


# Compat: mantenemos el nombre anterior usado por la UI
def balance_8v8(players: Sequence[Player], config: Dict[str, Any]) -> TeamResult:
    return balance_teams(players, config, objective_mode="weighted")

