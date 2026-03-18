from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Player:
    name: str
    football: float
    physical: float
    pos1: str
    pos2: str


DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": {"physical": 0.35, "football": 0.65},
    "team_size": 8,
    # Si el Excel trae físico categórico (Bajo/Medio/Alto), se normaliza a número.
    # Ajustable sin tocar el Excel.
    "physical_level_map": {
        "bajo": 1.0,
        "medio": 2.0,
        "alto": 3.0,
    },
    "positional_rules": {
        "GK": {"min": 0, "max": 1},
        "DEF": {"min": 2, "max": 3},
        "MID": {"min": 2, "max": 3},
        "FWD": {"min": 1, "max": 2},
    },
    # Reglas por formato (7v7 / 8v8). Si existe, se prioriza sobre positional_rules.
    "positional_rules_by_team_size": {
        "8": {
            "GK": {"min": 0, "max": 1},
            "DEF": {"min": 3, "max": 3},
            "MID": {"min": 3, "max": 3},
            "FWD": {"min": 1, "max": 2},
        },
        "7": {
            "GK": {"min": 0, "max": 1},
            "DEF": {"min": 2, "max": 3},
            "MID": {"min": 2, "max": 3},
            "FWD": {"min": 1, "max": 2},
        },
    },
    # Mapeo base (editable en el futuro) de etiquetas del Excel -> roles internos
    "position_map": {
        "portero": "GK",
        "defensa": "DEF",
        "medio": "MID",
        "delantero": "FWD",
        "atacante": "FWD",
    },
}


def _slug(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _coerce_float(x: Any) -> float:
    if pd.isna(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def ensure_data_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)


def config_paths(data_dir: Path) -> Dict[str, Path]:
    return {
        "players_csv": data_dir / "players.csv",
        "config_json": data_dir / "config.json",
        "history_json": data_dir / "history.json",
        "learning_json": data_dir / "learning.json",
    }


def load_or_init_config(data_dir: Path) -> Dict[str, Any]:
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["config_json"]
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        merged = _merge_dicts(DEFAULT_CONFIG, cfg)
        # Migración suave: si venías del mapping antiguo, actualizamos a la nueva escala por defecto
        old_map = {"bajo": 2.0, "medio": 3.0, "alto": 4.0}
        if merged.get("physical_level_map") in (None, old_map):
            merged["physical_level_map"] = dict(DEFAULT_CONFIG["physical_level_map"])
        # Asegurar atacante -> delantero si no está
        merged.setdefault("position_map", {})
        if "atacante" not in merged["position_map"]:
            merged["position_map"]["atacante"] = "FWD"

        # Migración suave de pesos: si tenías el default anterior (0.40/0.60), pasamos a (0.35/0.65)
        w = merged.get("weights", {}) or {}
        phys = float(w.get("physical", 0.0))
        foot = float(w.get("football", 0.0))
        if abs(phys - 0.40) < 1e-9 and abs(foot - 0.60) < 1e-9:
            merged["weights"] = {"physical": 0.35, "football": 0.65}
        return merged
    except Exception:
        # Si el JSON se corrompe, fallback seguro
        return dict(DEFAULT_CONFIG)


def save_config(data_dir: Path, config: Dict[str, Any]) -> None:
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["config_json"]
    p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_players_from_excel(excel_path: Path) -> pd.DataFrame:
    """
    Lee el Excel original (solo lectura) y devuelve un DataFrame normalizado a columnas internas:
    - name, football, physical, pos1, pos2
    """
    df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")
    colmap = {_slug(c): c for c in df.columns}

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in colmap:
                return colmap[cand]
        return None

    name_col = pick("persona", "nombre", "name")
    football_col = pick("nivel_futbolistico", "nivel_futbolstico", "futbol", "football")
    physical_col = pick("nivel_fisico", "nivel_fsico", "fisico", "physical")
    pos1_col = pick("posicion_1", "posicin_1", "pos1")
    pos2_col = pick("posicion_2", "posicin_2", "pos2")

    missing = [k for k, v in {
        "name": name_col,
        "football": football_col,
        "physical": physical_col,
        "pos1": pos1_col,
        "pos2": pos2_col,
    }.items() if v is None]
    if missing:
        raise ValueError(
            "No se pudieron detectar columnas en el Excel. Faltan: "
            + ", ".join(missing)
            + f". Columnas detectadas: {list(df.columns)}"
        )

    # físico puede venir numérico o como niveles (Bajo/Medio/Alto)
    phys_series = df[physical_col]
    if getattr(phys_series, "dtype", None) == object:
        # usamos el default aquí; en la app se persiste/edita en config.json y el CSV queda numérico
        level_map = DEFAULT_CONFIG.get("physical_level_map", {})
        phys_vals = phys_series.astype(str).str.strip()
        phys_num = phys_vals.map(lambda v: level_map.get(_slug(v), 0.0))
    else:
        phys_num = phys_series.map(_coerce_float)

    out = pd.DataFrame({
        "name": df[name_col].astype(str).str.strip(),
        "football": df[football_col].map(_coerce_float),
        "physical": phys_num,
        "pos1": df[pos1_col].astype(str).str.strip(),
        "pos2": df[pos2_col].astype(str).fillna("").astype(str).str.strip(),
    })

    out = out.dropna(subset=["name"])
    out = out[out["name"].str.len() > 0]
    out = out.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    return out


def load_or_init_players(data_dir: Path, excel_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fuente de verdad: `players.csv`.
    Si no existe y se provee excel_path, se inicializa desde el Excel.
    """
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["players_csv"]
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8")
        return _normalize_players_df(df)

    if excel_path is None:
        df = pd.DataFrame(columns=["name", "football", "physical", "pos1", "pos2"])
        df.to_csv(p, index=False, encoding="utf-8")
        return df

    df = load_players_from_excel(excel_path)
    df = _normalize_players_df(df)
    df.to_csv(p, index=False, encoding="utf-8")
    return df


def save_players(data_dir: Path, players_df: pd.DataFrame) -> None:
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["players_csv"]
    df = _normalize_players_df(players_df)
    df.to_csv(p, index=False, encoding="utf-8")


def _normalize_players_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["name", "football", "physical", "pos1", "pos2"]
    out = df.copy()
    for col in needed:
        if col not in out.columns:
            out[col] = "" if col in ("name", "pos1", "pos2") else 0.0
    out["name"] = out["name"].astype(str).str.strip()
    out["pos1"] = out["pos1"].astype(str).str.strip()
    out["pos2"] = out["pos2"].astype(str).fillna("").astype(str).str.strip()
    out["football"] = out["football"].map(_coerce_float)
    out["physical"] = out["physical"].map(_coerce_float)
    out = out[out["name"].str.len() > 0]
    out = out.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    return out[needed]


def df_to_players(df: pd.DataFrame) -> Tuple[Player, ...]:
    df = _normalize_players_df(df)
    return tuple(
        Player(
            name=str(r["name"]),
            football=float(r["football"]),
            physical=float(r["physical"]),
            pos1=str(r["pos1"]),
            pos2=str(r["pos2"]),
        )
        for _, r in df.iterrows()
    )


def load_history(data_dir: Path) -> Dict[str, Any]:
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["history_json"]
    if not p.exists():
        p.write_text(json.dumps({"events": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"events": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"events": []}


def append_history_event(data_dir: Path, event: Dict[str, Any]) -> None:
    h = load_history(data_dir)
    h.setdefault("events", [])
    h["events"].append(event)
    p = config_paths(data_dir)["history_json"]
    p.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")


def load_learning(data_dir: Path) -> Dict[str, Any]:
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["learning_json"]
    if not p.exists():
        p.write_text(json.dumps({"swap_counts": {}, "move_counts": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"swap_counts": {}, "move_counts": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"swap_counts": {}, "move_counts": {}}


def save_learning(data_dir: Path, learning: Dict[str, Any]) -> None:
    ensure_data_dir(data_dir)
    p = config_paths(data_dir)["learning_json"]
    p.write_text(json.dumps(learning, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute_learning_from_history(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aprendizaje heurístico simple:
    - move_counts[player]: cuántas veces se movió manualmente
    - swap_counts["A||B"]: cuántas veces un swap implicó (A,B) en recomendaciones aceptadas
      (por ahora inferimos swaps como dos movimientos consecutivos en sentido opuesto dentro de la misma sesión)
    """
    events = history.get("events", [])
    move_counts: Dict[str, int] = {}
    swap_counts: Dict[str, int] = {}

    for ev in events:
        if ev.get("type") == "move":
            n = str(ev.get("player", "")).strip()
            if n:
                move_counts[n] = move_counts.get(n, 0) + 1

    # Heurística: detectar swaps simples por pareja dentro de una "action_group"
    grouped: Dict[str, list] = {}
    for ev in events:
        gid = str(ev.get("group_id", ""))
        if not gid:
            continue
        grouped.setdefault(gid, []).append(ev)

    for gid, evs in grouped.items():
        moves = [e for e in evs if e.get("type") == "move"]
        if len(moves) < 2:
            continue
        # si hay dos jugadores movidos en direcciones opuestas, contarlo como swap
        for i in range(len(moves) - 1):
            a = moves[i]
            b = moves[i + 1]
            if a.get("from_team") == b.get("to_team") and a.get("to_team") == b.get("from_team"):
                p1 = str(a.get("player", "")).strip()
                p2 = str(b.get("player", "")).strip()
                if p1 and p2 and p1 != p2:
                    key = "||".join(sorted([p1, p2]))
                    swap_counts[key] = swap_counts.get(key, 0) + 1

    return {"swap_counts": swap_counts, "move_counts": move_counts}

