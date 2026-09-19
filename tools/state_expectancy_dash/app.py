"""Strategery dashboard: interactive win probability, RE24, and transition explorer."""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import duckdb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------- #
# Copied over from `projects/strategery/constants.py`.
# Yes it's redundant, but easier to containerize so
# worth it for the duped code.
# roll it after this many runs
MAX_RUNS_PER_INNING = 15

# game's over when score gap gets this bad
MAX_SCORE_DIFFERENTIAL = 15

# array of runs you can score in an inning
RUNS_ARRAY = np.arange(MAX_RUNS_PER_INNING + 1)
# ----------------------------------------------------- #

# column names for probability of runs scored
PROB_RS_COLS = ["p_" + str(item) for item in RUNS_ARRAY]

# home leads are only modeled out to +/- (MAX_SCORE_DIFFERENTIAL - 1); anything beyond gets clipped here.
MAX_HOME_LEAD = MAX_SCORE_DIFFERENTIAL - 1


# clip run-scoring distribution display at this many runs (everything above rolls into a "10+" bucket)
RUNS_DISPLAY_CAP = 8

# flip this if you need to debug and view the session state in real time
DEBUG = False

# again, doing a copy so that these can be pushed to GH in a clean / separate space.
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
STATIC_PARQUETS = dict(
    re24=os.path.join(DATA_DIR, "re24", "re24.parquet"),
    re24_transition_probs=os.path.join(DATA_DIR, "re24_transition_probs", "re24_transition_probs.parquet"),
    win_probability=os.path.join(DATA_DIR, "win_probability", "win_probability.parquet"),
    leverage_index=os.path.join(DATA_DIR, "leverage_index", "leverage_index.parquet"),
    future_leverage=os.path.join(DATA_DIR, "future_leverage", "future_leverage.parquet"),
)


# ----------------------------------------------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------------------------------------------- #
def query(sql: str) -> pd.DataFrame:
    """
    Helper to query from DuckDB. Copied from utils/duckdb.py

    Parameters
    ----------
    sql : str
        SQL-like code to hit DuckDB

    Returns
    -------
    pd.DataFrame
        The dataframe queried via duckdb.
    """
    con = duckdb.connect()
    df = con.execute(sql).df()
    return df


@st.cache_data
def load_re24() -> pd.DataFrame:
    """
    Load the RE24 table, via duckdb, from the app's data folder.

    Returns
    -------
    pd.DataFrame
        RE24 values with run probability distributions by game state and half inning.
    """
    return query(f"SELECT * FROM '{STATIC_PARQUETS['re24']}'")


@st.cache_data
def load_transition_probs() -> pd.DataFrame:
    """
    Load the PA-level transition probabilities, via duckdb, from the app's data folder.

    Returns
    -------
    pd.DataFrame
        Transition probabilities by inning_topbot, game_state, game_state_post, runs.
    """
    return query(f"SELECT * FROM '{STATIC_PARQUETS['re24_transition_probs']}'")


@st.cache_data
def load_win_probs() -> pd.DataFrame:
    """
    Load the simulated/closed-form win probability table, via duckdb,
    from the app's data folder.

    Returns
    -------
    pd.DataFrame
        Win probabilities by game_state, inning, inning_topbot, home_lead
    """
    return query(f"SELECT * FROM '{STATIC_PARQUETS['win_probability']}'")


@st.cache_data
def load_leverage() -> pd.DataFrame:
    """
    Load leverage indices.

    Returns
    -------
    pd.DataFrame
        Leverage indices by game_state, inning, inning_topbot, home_lead.
    """
    sql = f"""
    SELECT
        li.game_state,
        li.inning,
        li.inning_topbot,
        li.home_lead,
        li.leverage_index,
        fl.expected_visits_1_plus li_1,
        fl.expected_visits_2_plus li_2,
        fl.expected_visits_3_plus li_3,
        fl.expected_visits_4_plus li_4
    FROM '{STATIC_PARQUETS['leverage_index']}' li
    LEFT JOIN '{STATIC_PARQUETS['future_leverage']}' fl
        USING(game_state, inning, inning_topbot, home_lead)
    """
    return query(sql)


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


def clip_home_lead(home_lead: int) -> tuple[int, bool]:
    """
    Clip a home lead to the modeled range and report whether clipping occurred.

    The win probability model only covers leads within +/- MAX_HOME_LEAD; anything
    beyond that is treated as a decided game and clipped to the boundary.

    Parameters
    ----------
    home_lead : int
        Raw home lead (home score minus away score).

    Returns
    -------
    clipped_lead : int
        Home lead clamped to [-MAX_HOME_LEAD, MAX_HOME_LEAD].
    was_clipped : bool
        True if the input fell outside the modeled range and was clamped.
    """
    clipped_lead = int(np.clip(home_lead, -MAX_HOME_LEAD, MAX_HOME_LEAD))
    return clipped_lead, clipped_lead != home_lead


# ----------------------------------------------------------------------------------------------------------------- #
# Diamond rendering
# ----------------------------------------------------------------------------------------------------------------- #
def linebreak() -> None:
    """Inserts a narrow linebreak, with less padding than st.divide()"""
    st.markdown(
        "<hr style='margin: 0.2rem 0; border: none; border-top: 1px solid #ddd;' />",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------------------------------------------- #
# Diamond rendering
# ----------------------------------------------------------------------------------------------------------------- #


def _toggle(base: str) -> None:
    """Flip a base's occupancy in session_state (runs before the rerun)."""
    st.session_state[base] = not st.session_state[base]


def render_diamond() -> None:
    """
    Render a clickable baseball diamond that toggles base occupancy in session_state.

    Uses on_click callbacks so the toggle lands before the rerun, keeping button colors
    and downstream game_state in lockstep on a single click. Layout mimics a diamond:
    2B on top, 3B left / 1B right, home at the bottom.
    """

    def _label(occupied: bool, name: str) -> str:
        return f"🟢 {name}" if occupied else f"⚪ {name}"

    # 2nd base row (top of the diamond)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.button(
            _label(st.session_state.on_second, "2B"),
            use_container_width=True,
            key="btn_2b",
            on_click=_toggle,
            args=("on_second",),
        )

    # 3rd / 1st base row (middle of the diamond)
    left, _, right = st.columns([1, 1, 1])
    with left:
        st.button(
            _label(st.session_state.on_third, "3B"),
            use_container_width=True,
            key="btn_3b",
            on_click=_toggle,
            args=("on_third",),
        )
    with right:
        st.button(
            _label(st.session_state.on_first, "1B"),
            use_container_width=True,
            key="btn_1b",
            on_click=_toggle,
            args=("on_first",),
        )

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

    The run-scoring distribution is displayed with everything at or above
    RUNS_DISPLAY_CAP collapsed into a single "10+" bucket.

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

    # raw per-run probabilities, then fold everything >= RUNS_DISPLAY_CAP into a single bucket
    probs = row[PROB_RS_COLS].values.flatten()
    capped_probs = probs[:RUNS_DISPLAY_CAP].tolist() + [probs[RUNS_DISPLAY_CAP:].sum()]
    capped_labels = [str(r) for r in range(RUNS_DISPLAY_CAP)] + [f"{RUNS_DISPLAY_CAP}+"]

    dist = pd.DataFrame({"runs": capped_labels, "probability": capped_probs})

    chart = (
        alt.Chart(dist)
        .mark_bar()
        .encode(
            # ordinal (:O) treats runs as discrete categories; sort keeps 0..10+ in order; labelAngle=0 keeps upright
            x=alt.X("runs:O", title="Runs scored", sort=capped_labels, axis=alt.Axis(labelAngle=0)),
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
        Home score minus away score (already clipped to the modeled range).
    """
    row = win_probs.query(
        "game_state == @game_state & inning_topbot == @half_inning & inning == @inning & home_lead == @home_lead"
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


def report_leverage(leverage: pd.DataFrame, game_state: str, half_inning: str, inning: int, home_lead: int) -> None:
    """
    Display the leverage index for the selected state.

    Parameters
    ----------
    leverage : pd.DataFrame
        The loaded leverage index table.
    game_state : str
        Canonical game state string.
    half_inning : str
        'Top' or 'Bot'.
    inning : int
        Current inning number.
    home_lead : int
        Home score minus away score (already clipped to the modeled range).
    """
    row = leverage.query(
        "game_state == @game_state & inning_topbot == @half_inning & inning == @inning & home_lead == @home_lead"
    )
    if row.empty:
        st.warning(
            f"No leverage record for {game_state} ({half_inning}, inning {inning}, lead {home_lead}). "
            "This state may have been pruned as impossible."
        )
        return

    # LI metric on the left, 1x4 expected-visits table on the right
    li_col, li_fut_table = st.columns([1, 3])

    # plot the current leverage
    with li_col:
        li = float(row["leverage_index"].iloc[0])
        st.metric("Leverage Index", f"{li:.2f}")

    with li_fut_table:
        lev_thresholds = [1, 2, 3, 4]
        visit_cols = [f"li_{item}" for item in lev_thresholds]
        table_map = {f"li_{item}": f"E[PA w/ LI ≥ {item}]" for item in lev_thresholds}
        # rename to friendly headers and format to 1 decimal
        table = row[visit_cols].rename(columns=table_map).round(1)
        st.caption("Expected rest-of-game PAs above leverage threshold")
        st.dataframe(table.reset_index(drop=True), use_container_width=True, hide_index=True)


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
        .rename(columns={"game_state_post": "Next State", "runs": "Runs Scored", "prob": "Probability"})
    )
    if trans.empty:
        st.warning(f"No transitions for {game_state} ({half_inning}).")
        return

    trans["Probability"] = trans["Probability"].map("{:.1%}".format)

    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] div[role="gridcell"] {
            justify-content: center;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(trans.reset_index(drop=True), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------------------------------------------- #
# App
# ----------------------------------------------------------------------------------------------------------------- #
def main() -> None:
    """
    Run the Strategery dashboard.

    Provides a clickable diamond for setting runners, controls for score/outs/inning,
    and reports win probability, leverage, RE24, and the most likely state transitions.
    """
    st.set_page_config(page_title="Strategery Dashboard", layout="wide")

    # # title
    # st.markdown(
    #     "<h1 style='text-align: center;'>Strategery Navigator</h1>",
    #     unsafe_allow_html=True,
    # )
    # linebreak()

    # initialize base occupancy state
    for base in ("on_first", "on_second", "on_third"):
        if base not in st.session_state:
            st.session_state[base] = False

    re24 = load_re24()
    transition_probs = load_transition_probs()
    win_probs = load_win_probs()
    leverage = load_leverage()

    # ---- controls ---- #
    left, right = st.columns([1, 2], gap="small", border=True)

    with left:
        st.subheader("Game State")
        render_diamond()

        outs = st.radio("Outs", options=[0, 1, 2], horizontal=True, key="outs")

        inning_col, half_col = st.columns(2)
        inning = inning_col.number_input("Inning", min_value=1, max_value=10, value=1, step=1, key="inning")
        half_inning = half_col.radio("Half", options=["Top", "Bot"], horizontal=True, key="half_inning")

        score_col1, score_col2 = st.columns(2)
        home_score = score_col1.number_input("Home score", min_value=0, value=0, step=1, key="home_score")
        away_score = score_col2.number_input("Away score", min_value=0, value=0, step=1, key="away_score")
        home_lead = int(home_score - away_score)

    # clip the lead into the modeled range; warn if we had to
    home_lead, lead_was_clipped = clip_home_lead(home_lead)

    game_state = build_game_state(
        st.session_state.on_first,
        st.session_state.on_second,
        st.session_state.on_third,
        outs,
    )
    if DEBUG:
        st.sidebar.write("DEBUG session_state:", dict(st.session_state))

    with right:
        if lead_was_clipped:
            st.warning(
                f"Games are capped at a +/- {MAX_HOME_LEAD}-run differential; the lead has been clamped to "
                f"{home_lead:+d} for these calculations."
            )

        st.markdown("### Win Probability")
        report_win_probability(win_probs, game_state, half_inning, int(inning), home_lead)
        linebreak()

        st.markdown("### Leverage")
        report_leverage(leverage, game_state, half_inning, int(inning), home_lead)
        st.markdown(
            "<hr style='margin: 0.3rem 0; border: none; border-top: 1px solid #ddd;' />",
            unsafe_allow_html=True,
        )

        st.markdown("### Run Expectancy")
        report_run_expectancy(re24, game_state, half_inning)
        linebreak()

        st.markdown("### Most Likely Transitions")
        report_transitions(transition_probs, game_state, half_inning)


if __name__ == "__main__":
    main()
