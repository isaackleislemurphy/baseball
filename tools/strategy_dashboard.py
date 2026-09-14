"""Strategery dashboard: interactive win probability, RE24, and transition explorer."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from baseball.duckdb.database import TABLES
from baseball.projects.strategery.constants import RUNS_ARRAY
from baseball.utils.duckdb import query

PROB_RS_COLS = ["p_" + str(item) for item in RUNS_ARRAY]


# ----------------------------------------------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------------------------------------------- #
@st.cache_data
def load_re24() -> pd.DataFrame:
    """
    Load the RE24 table from DuckDB.

    Returns
    -------
    pd.DataFrame
        RE24 values with run probability distributions by game state and half inning.
    """
    return query(f"SELECT * FROM '{TABLES.strategery.re24}'")


@st.cache_data
def load_transition_probs() -> pd.DataFrame:
    """
    Load the PA-level transition probabilities from DuckDB.

    Returns
    -------
    pd.DataFrame
        Transition probabilities by inning_topbot, game_state, game_state_post, runs, prob.
    """
    return query(f"SELECT * FROM '{TABLES.strategery.re24_transition_probs}'")


@st.cache_data
def load_win_probs() -> pd.DataFrame:
    """
    Load the simulated/closed-form win probability table from DuckDB.

    Returns
    -------
    pd.DataFrame
        Win probabilities by game_state, inning, inning_topbot, home_lead, home_win_prob.
    """
    return query(f"SELECT * FROM '{TABLES.strategery.win_probability}'")


# ----------------------------------------------------------------------------------------------------------------- #
# State helpers
# ----------------------------------------------------------------------------------------------------------------- #
def build_game_state(on_first: bool, on_second: bool, on_third: bool, outs: int) -> str:
    """
    Assemble a canonical game_state string from base/out components.

    Parameters
    ----------
    on_first, on_second, on_third : bool
        Whether each base is occupied.
    outs : int
        Number of outs (0, 1, or 2).

    Returns
    -------
    str
        Game state formatted like '1-3:2' (bases then colon then outs).
    """
    bases = ("1" if on_first else "-") + ("2" if on_second else "-") + ("3" if on_third else "-")
    return f"{bases}:{outs}"


# ----------------------------------------------------------------------------------------------------------------- #
# Diamond rendering
# ----------------------------------------------------------------------------------------------------------------- #
def render_diamond(on_first: bool, on_second: bool, on_third: bool) -> None:
    """
    Render a clickable baseball diamond using Streamlit buttons.

    Toggles the occupancy of each base in session_state when its button is clicked.
    Layout mimics a diamond: 2B on top, 3B left / 1B right, home at the bottom.

    Parameters
    ----------
    on_first, on_second, on_third : bool
        Current occupancy of each base (used to label the buttons).
    """

    def _label(occupied: bool, name: str) -> str:
        return f"🟢 {name}" if occupied else f"⚪ {name}"

    # 2nd base row (top of the diamond)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button(_label(on_second, "2B"), use_container_width=True, key="btn_2b"):
            st.session_state.on_second = not st.session_state.on_second
            st.rerun()

    # 3rd / 1st base row (middle of the diamond)
    left, _, right = st.columns([1, 1, 1])
    with left:
        if st.button(_label(on_third, "3B"), use_container_width=True, key="btn_3b"):
            st.session_state.on_third = not st.session_state.on_third
            st.rerun()
    with right:
        if st.button(_label(on_first, "1B"), use_container_width=True, key="btn_1b"):
            st.session_state.on_first = not st.session_state.on_first
            st.rerun()

    # home plate row (bottom of the diamond)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.button("Home", use_container_width=True, disabled=True, key="btn_home")


# ----------------------------------------------------------------------------------------------------------------- #
# Reports
# ----------------------------------------------------------------------------------------------------------------- #
def report_run_expectancy(re24: pd.DataFrame, game_state: str, half_inning: str) -> None:
    """
    Display the RE24 and run-distribution for the selected state and half inning.

    Parameters
    ----------
    re24 : pd.DataFrame
        The loaded RE24 table.
    game_state : str
        Canonical game state string (e.g., '1-3:2').
    half_inning : str
        'Top' or 'Bot'.
    """
    row = re24.query("game_state == @game_state & inning_topbot == @half_inning")
    if row.empty:
        st.warning(f"No RE24 record for {game_state} ({half_inning}).")
        return

    run_exp = float(row["run_exp"].iloc[0])
    st.metric("Run Expectancy (RE24)", f"{run_exp:.3f}")

    dist = pd.DataFrame(
        {
            "runs": RUNS_ARRAY,
            "probability": row[PROB_RS_COLS].values.flatten(),
        }
    )

    chart = (
        alt.Chart(dist)
        .mark_bar()
        .encode(
            # ordinal (:O) treats runs as discrete categories; labelAngle=0 keeps them upright
            x=alt.X("runs:O", title="Runs scored", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("probability:Q", title="Probability"),
        )
        .properties(height=200)  # <-- shorter chart; tweak to taste
    )
    st.altair_chart(chart, use_container_width=True)


def report_win_probability(
    win_probs: pd.DataFrame, game_state: str, half_inning: str, inning: int, home_lead: int
) -> None:
    """
    Display the home win probability for the selected state.

    Parameters
    ----------
    win_probs : pd.DataFrame
        The loaded win probability table.
    game_state : str
        Canonical game state string.
    half_inning : str
        'Top' or 'Bot'.
    inning : int
        Current inning number.
    home_lead : int
        Home score minus away score.
    """
    row = win_probs.query(
        "game_state == @game_state & inning_topbot == @half_inning " "& inning == @inning & home_lead == @home_lead"
    )
    if row.empty:
        st.warning(
            f"No win-prob record for {game_state} ({half_inning}, inning {inning}, lead {home_lead}). "
            "This state may have been pruned as impossible."
        )
        return

    hwp = float(row["home_win_prob"].iloc[0])
    col1, col2 = st.columns(2)
    col1.metric("Home Win Probability", f"{hwp:.1%}")
    col2.metric("Away Win Probability", f"{1 - hwp:.1%}")


def report_transitions(transition_probs: pd.DataFrame, game_state: str, half_inning: str, top_n: int = 8) -> None:
    """
    Display the most likely next-state transitions from the current state.

    Parameters
    ----------
    transition_probs : pd.DataFrame
        The loaded transition probabilities table.
    game_state : str
        Canonical game state string.
    half_inning : str
        'Top' or 'Bot'.
    top_n : int, optional
        Number of transitions to show, by default 8.
    """
    trans = (
        transition_probs.query("game_state == @game_state & inning_topbot == @half_inning")
        .sort_values("prob", ascending=False)
        .head(top_n)
        .loc[:, ["game_state_post", "runs", "prob"]]
        .rename(columns={"game_state_post": "next_state", "runs": "runs_scored", "prob": "probability"})
    )
    if trans.empty:
        st.warning(f"No transitions for {game_state} ({half_inning}).")
        return

    trans["probability"] = trans["probability"].map("{:.1%}".format)
    st.dataframe(trans.reset_index(drop=True), use_container_width=True)


# ----------------------------------------------------------------------------------------------------------------- #
# App
# ----------------------------------------------------------------------------------------------------------------- #
def main() -> None:
    """
    Run the Strategery dashboard.

    Provides a clickable diamond for setting runners, controls for score/outs/inning,
    and reports win probability, RE24, and the most likely state transitions.
    """
    st.set_page_config(page_title="Strategery Dashboard", layout="wide")

    # title
    st.markdown(
        "<h1 style='text-align: center;'>Strategery Navigator</h1>",
        unsafe_allow_html=True,
    )
    st.divider()  # the bar separator; or use st.markdown("---")

    # initialize base occupancy state
    for base in ("on_first", "on_second", "on_third"):
        if base not in st.session_state:
            st.session_state[base] = False

    re24 = load_re24()
    transition_probs = load_transition_probs()
    win_probs = load_win_probs()

    # ---- controls ---- #
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Game State")
        render_diamond(
            st.session_state.on_first,
            st.session_state.on_second,
            st.session_state.on_third,
        )

        # st.subheader("Situation")
        outs = st.radio("Outs", options=[0, 1, 2], horizontal=True)
        inning = st.number_input("Inning", min_value=1, max_value=10, value=1, step=1)
        half_inning = st.radio("Half", options=["Top", "Bot"], horizontal=True)

        # st.subheader("Score")
        # home_score = st.number_input("Home score", min_value=0, value=0, step=1)
        # away_score = st.number_input("Away score", min_value=0, value=0, step=1)
        # home_lead = int(home_score - away_score)

        # st.subheader("Score")
        score_col1, score_col2 = st.columns(2)
        home_score = score_col1.number_input("Home score", min_value=0, value=0, step=1)
        away_score = score_col2.number_input("Away score", min_value=0, value=0, step=1)
        home_lead = int(home_score - away_score)

    game_state = build_game_state(
        st.session_state.on_first,
        st.session_state.on_second,
        st.session_state.on_third,
        outs,
    )

    with right:
        # sst.subheader(f"Current state: `{game_state}` — {half_inning} {inning}, home lead {home_lead:+d}")

        st.markdown("### Win Probability")
        report_win_probability(win_probs, game_state, half_inning, int(inning), home_lead)

        st.markdown("### Run Expectancy")
        report_run_expectancy(re24, game_state, half_inning)

        st.markdown("### Most Likely Transitions")
        report_transitions(transition_probs, game_state, half_inning)


if __name__ == "__main__":
    main()
