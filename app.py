from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

import data_manager as dm
from recommendation_engine import recommend_swaps_by_metric
from team_balancer import TeamResult, balance_2_options


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def _load_state(excel_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = dm.load_or_init_config(DATA_DIR)
    players_df = dm.load_or_init_players(DATA_DIR, excel_path=excel_path)
    # Historial/aprendizaje desactivado por UX (se puede reactivar más adelante)
    return players_df, cfg, {}, {}


def _df_to_players_for_selection(df: pd.DataFrame, selected_names: List[str]) -> Tuple[dm.Player, ...]:
    df = df[df["name"].isin(selected_names)].copy()
    return dm.df_to_players(df)


def _top4_split_status(players: Tuple[dm.Player, ...], team_a: Tuple[dm.Player, ...]) -> Dict[str, str]:
    # Top 4 por fútbol y físico (2 y 2 ideal)
    top_k = 4
    top_f = [p.name for p in sorted(players, key=lambda p: p.football, reverse=True)[:top_k]]
    top_p = [p.name for p in sorted(players, key=lambda p: p.physical, reverse=True)[:top_k]]
    set_a = set(p.name for p in team_a)
    cnt_f_a = sum(1 for n in top_f if n in set_a)
    cnt_p_a = sum(1 for n in top_p if n in set_a)
    return {
        "Top4 fútbol (A/B)": f"{cnt_f_a}/{top_k-cnt_f_a}",
        "Top4 físico (A/B)": f"{cnt_p_a}/{top_k-cnt_p_a}",
    }


st.set_page_config(page_title="Muga Fútbol - Generador de Alineaciones", layout="wide")
st.title("Muga Fútbol - Generador de Alineaciones")

excel_path = st.sidebar.text_input(
    "Ruta del Excel (solo lectura)",
    value=str((APP_DIR / "Fulbichi.xlsx").resolve()),
)
excel_p = Path(excel_path)

try:
    players_df, cfg, _history, _learning = _load_state(excel_p)
except Exception as e:
    st.error(f"No pude cargar datos: {e}")
    st.stop()

st.sidebar.header("Configuración")

PRESELECT_NAMES = ["Kike", "Migue"]
default_sel = players_df["name"].tolist()[:16]
for nm in PRESELECT_NAMES:
    if nm in players_df["name"].tolist() and nm not in default_sel:
        # Sustituimos un jugador "normal" por el preseleccionado para mantener tamaño 16.
        for i in range(len(default_sel) - 1, -1, -1):
            if default_sel[i] not in PRESELECT_NAMES:
                default_sel[i] = nm
                break
st.session_state.setdefault("selected_names", default_sel)
st.session_state.setdefault("result_opt1", None)
st.session_state.setdefault("result_opt2", None)
st.session_state.setdefault("has_generated", False)


def _recalc_if_possible() -> None:
    selected = st.session_state.get("selected_names", [])
    if not isinstance(selected, list) or len(selected) not in (14, 16):
        return
    if not st.session_state.get("has_generated", False):
        return
    cfg_local = st.session_state.get("cfg")
    df_local = st.session_state.get("players_df")
    if not isinstance(cfg_local, dict) or not isinstance(df_local, pd.DataFrame):
        return
    players_local = _df_to_players_for_selection(df_local, selected)
    cfg_local["team_size"] = len(selected) // 2
    r1, r2 = balance_2_options(players_local, cfg_local, objective_mode="weighted")
    st.session_state["result_opt1"] = r1
    st.session_state["result_opt2"] = r2


w_phys = st.sidebar.slider(
    "Ponderación físico",
    0.0,
    1.0,
    0.35,
    0.05,
    on_change=_recalc_if_possible,
    key="w_phys",
)
w_foot = 1.0 - float(w_phys)
st.sidebar.write(f"Ponderación fútbol: **{w_foot:.2f}**")
cfg["weights"]["physical"] = float(w_phys)
cfg["weights"]["football"] = float(w_foot)

if st.sidebar.button("Guardar configuración", use_container_width=True):
    dm.save_config(DATA_DIR, cfg)
    st.sidebar.success("Configuración guardada.")

st.session_state["cfg"] = cfg
st.session_state["players_df"] = players_df

st.divider()

tab1, tab2 = st.tabs(["Partido", "Jugadores (CRUD)"])

with tab2:
    st.subheader("Editar base de jugadores (persistencia: `data/players.csv`)")
    st.caption("Tip: el Excel se usa solo para inicializar. Desde aquí puedes editar sin riesgo.")

    edited = st.data_editor(
        players_df,
        use_container_width=True,
        num_rows="dynamic",
        key="players_editor",
        column_config={
            "name": st.column_config.TextColumn("Nombre", required=True),
            "football": st.column_config.NumberColumn("Fútbol", min_value=0.0, max_value=10.0, step=0.1),
            "physical": st.column_config.NumberColumn("Físico", min_value=0.0, max_value=10.0, step=0.1),
            "pos1": st.column_config.TextColumn("Posición 1"),
            "pos2": st.column_config.TextColumn("Posición 2"),
        },
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Guardar cambios de jugadores", type="primary", use_container_width=True):
        dm.save_players(DATA_DIR, edited)
        st.success("Guardado en `data/players.csv`.")
        players_df = edited

    if c2.button("Reinicializar desde Excel (sobrescribe CSV)", use_container_width=True):
        df = dm.load_players_from_excel(excel_p)
        dm.save_players(DATA_DIR, df)
        st.warning("CSV reinicializado desde el Excel.")
        players_df = df

with tab1:
    st.subheader("Convocatoria (selección rápida)")
    st.caption(
        "Selecciona con **multiselect**. Tip: usa el panel **Añadir en lote** para ir marcando varios y luego añadirlos de una vez. "
        "Deben ser **14 (7v7)** o **16 (8v8)**."
    )

    all_names = players_df["name"].tolist()

    st.session_state.setdefault("selected_names", players_df["name"].tolist()[:16])
    st.session_state.setdefault("batch_add", [])

    def _add_batch() -> None:
        cur = set(st.session_state.get("selected_names", []))
        cur |= set(st.session_state.get("batch_add", []))
        st.session_state["selected_names"] = sorted(cur)
        st.session_state["batch_add"] = []

    def _clear_selection() -> None:
        st.session_state["selected_names"] = []
        st.session_state["batch_add"] = []

    # Selección final (fuente de verdad)
    st.multiselect(
        "Convocados (final)",
        options=all_names,
        default=st.session_state.get("selected_names", []),
        key="selected_names",
        help="Escribe para buscar. Aquí ves la selección final.",
    )

    # Añadir en lote (para reducir fricción cuando el dropdown se cierra en tu navegador)
    with st.popover("Añadir en lote", use_container_width=True):
        st.multiselect(
            "Jugadores a añadir",
            options=all_names,
            default=[],
            key="batch_add",
            help="Selecciona varios y pulsa 'Añadir seleccionados'.",
        )
        c1, c2 = st.columns(2)
        c1.button("Añadir seleccionados", type="primary", use_container_width=True, on_click=_add_batch)
        c2.button("Limpiar todo", use_container_width=True, on_click=_clear_selection)

    st.write(f"Convocados: **{len(st.session_state['selected_names'])}**")
    if len(st.session_state["selected_names"]) not in (14, 16):
        st.info("Ajusta la convocatoria a **14** o **16** para generar alineaciones.")
        st.stop()

    players = _df_to_players_for_selection(players_df, st.session_state["selected_names"])
    cfg["team_size"] = len(players) // 2

    # Importante para UX/Streamlit Cloud: NO calculamos automáticamente al cargar.
    # Solo se calcula tras pulsar "Generar alineaciones" o "Recalcular".

    st.divider()
    st.subheader("Generación")
    cgen, crec, _ = st.columns([1, 1, 1])
    if cgen.button("Generar alineaciones", type="primary", use_container_width=True):
        r1, r2 = balance_2_options(players, cfg, objective_mode="weighted")
        st.session_state["result_opt1"] = r1
        st.session_state["result_opt2"] = r2
        st.session_state["has_generated"] = True
    if crec.button("Recalcular", use_container_width=True):
        r1, r2 = balance_2_options(players, cfg, objective_mode="weighted")
        st.session_state["result_opt1"] = r1
        st.session_state["result_opt2"] = r2
        st.session_state["has_generated"] = True

    res1: TeamResult | None = st.session_state.get("result_opt1")
    res2: TeamResult | None = st.session_state.get("result_opt2")
    if not res1 or not res2:
        st.stop()

    def _top3(team: Tuple[dm.Player, ...], key: str) -> List[str]:
        if key == "football":
            arr = sorted(team, key=lambda p: p.football, reverse=True)
        else:
            arr = sorted(team, key=lambda p: p.physical, reverse=True)
        return [p.name for p in arr[:3]]

    def _render_option(title: str, res: TeamResult) -> None:
        team_size = len(res.team_a)
        avg_a_total = res.score_a / team_size
        avg_b_total = res.score_b / team_size
        avg_a_foot = res.football_a / team_size
        avg_b_foot = res.football_b / team_size
        avg_a_phys = res.physical_a / team_size
        avg_b_phys = res.physical_b / team_size

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Avg Total A ({title})", f"{avg_a_total:.2f}")
        k2.metric(f"Avg Total B ({title})", f"{avg_b_total:.2f}")
        k3.metric(f"Avg Fútbol (A/B)", f"{avg_a_foot:.2f} / {avg_b_foot:.2f}")
        k4.metric(f"Avg Físico (A/B)", f"{avg_a_phys:.2f} / {avg_b_phys:.2f}")

        split = _top4_split_status(players, tuple(res.team_a))
        s1, s2 = st.columns(2)
        s1.metric("Top4 fútbol", split["Top4 fútbol (A/B)"])
        s2.metric("Top4 físico", split["Top4 físico (A/B)"])

        t1, t2 = st.columns(2)
        with t1:
            st.caption("Top 3 fútbol A: " + ", ".join(_top3(res.team_a, "football")))
            st.caption("Top 3 físico A: " + ", ".join(_top3(res.team_a, "physical")))
        with t2:
            st.caption("Top 3 fútbol B: " + ", ".join(_top3(res.team_b, "football")))
            st.caption("Top 3 físico B: " + ", ".join(_top3(res.team_b, "physical")))

        left, right = st.columns(2)
        with left:
            st.subheader("Equipo A")
            for p in res.team_a:
                with st.expander(p.name, expanded=False):
                    st.write(f"Fútbol: **{p.football:.1f}**")
                    st.write(f"Físico: **{p.physical:.1f}**")
                    st.write(f"Posiciones: **{p.pos1} / {p.pos2}**")
        with right:
            st.subheader("Equipo B")
            for p in res.team_b:
                with st.expander(p.name, expanded=False):
                    st.write(f"Fútbol: **{p.football:.1f}**")
                    st.write(f"Físico: **{p.physical:.1f}**")
                    st.write(f"Posiciones: **{p.pos1} / {p.pos2}**")

        st.divider()
        st.subheader("Recomendaciones (compensar)")
        c_f, c_p = st.columns(2)
        with c_f:
            st.write("**Fútbol**")
            recs_f = recommend_swaps_by_metric(res, metric="football", top_k=5)
            if not recs_f:
                st.caption("Ya está muy compensado en fútbol.")
            for r in recs_f[:3]:
                st.write(f"- {r.swap_a.name} ↔ {r.swap_b.name}  | ajuste fútbol {r.improvement_points:.2f}")
        with c_p:
            st.write("**Físico**")
            recs_p = recommend_swaps_by_metric(res, metric="physical", top_k=5)
            if not recs_p:
                st.caption("Ya está muy compensado en físico.")
            for r in recs_p[:3]:
                st.write(f"- {r.swap_a.name} ↔ {r.swap_b.name}  | ajuste físico {r.improvement_points:.2f}")

    tabs = st.tabs(["Opción 1", "Opción 2"])
    with tabs[0]:
        _render_option("Opt1", res1)
    with tabs[1]:
        _render_option("Opt2", res2)

