import itertools
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from tqdm import tqdm

from baseball.duckdb.database import TABLES, write_parquet
from baseball.projects.strategery.constants import (
    GAME_STATES,
    MAX_SCORE_DIFFERENTIAL,
)
from baseball.utils.duckdb import query
from baseball.utils.logging import get_logger

warnings.filterwarnings("ignore", category=PerformanceWarning)
LOGGER = get_logger(__name__)


def query_transition_probs() -> pd.DataFrame:
    """
    Query PA-level state transition probabilities from DuckDB.

    Pulls the empirical plate-appearance transition probabilities produced by the
    RE24 pipeline, keyed by half inning and starting game state.

    Returns
    -------
    pd.DataFrame
        Transition probabilities with columns: inning_topbot, game_state,
        game_state_post, runs (runs scored on the transition), and prob
        (empirical probability of the transition).
    """

    sql = f"""
    SELECT
        inning_topbot,
        game_state,
        game_state_post,
        runs,
        prob
    FROM '{TABLES.strategery.re24_transition_probs}'
    """
    return query(sql)


def construct_game_transition_matrix(transition_probs: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Build a game-level absorbing Markov transition matrix over full game states.

    Expands the PA-level transition probabilities into a game-wide state space
    defined by (inning, half_inning, home_lead, game_state), plus two absorbing
    terminal states ("home_win", "home_loss"). Impossible states (e.g., the home
    team leading in the top of the 1st, or leading in the bottom of the 9th+) are
    pruned from the state space. Transitions are routed to terminal states when the
    game is decided by score differential, walk-off, or a shut-the-door final out;
    otherwise they advance the half inning or reflect an in-inning advancement.

    Parameters
    ----------
    transition_probs : pd.DataFrame
        PA-level transition probabilities, as returned by `query_transition_probs`.

    Returns
    -------
    P_full : np.ndarray
        A square, row-stochastic transition matrix over the expanded game state
        space. Row/column ordering matches `full_states`. The final two entries
        correspond to the terminal states ("home_win", "home_loss").
    full_states : list[tuple]
        The ordered list of states indexing `P_full`. Each state is a tuple of
        (inning, half_inning, home_lead, game_state); terminal states use
        (None, None, None, "home_win") and (None, None, None, "home_loss").
    """

    # ================================================================================================ #
    # [STEP 1] Define the state space
    # ================================================================================================ #
    innings = np.arange(1, 11)
    half_innings = ("Top", "Bot")
    home_leads = np.arange(-MAX_SCORE_DIFFERENTIAL + 1, MAX_SCORE_DIFFERENTIAL)  # [-MAX_SCORE_DIFF, MAX_SCORE_DIFF]
    full_states = tuple(itertools.product(innings, half_innings, home_leads, GAME_STATES))
    # lop off impossible starting states; as much as it might feel like it, home team can't be leading T1.
    full_states = tuple([item for item in full_states if not (item[0] == 1 and item[1] == "Top" and item[2] > 0)])
    # can't do better than a walk off
    full_states = tuple([item for item in full_states if not (item[0] >= 9 and item[1] == "Bot" and item[2] > 0)])
    # tack on terminal states
    full_states += ((None, None, None, "home_win"), (None, None, None, "home_loss"))

    # ================================================================================================ #
    # [STEP 2] Init empty transition matrix
    # ================================================================================================ #
    # NOTE—this is the first big/sparse Markov chain I've spun up in this repo, which means it's the first time I've
    # gotten burned trying to repeatedly insert into a massive dataframe. So at some point, the other Markov
    # matrices should be numpy-ified like this.
    state_to_idx = {state: i for i, state in enumerate(full_states)}
    P_full = np.zeros((len(full_states), len(full_states)))
    # terminal state self-transitions
    for term_state in ("home_win", "home_loss"):
        P_full[state_to_idx[(None, None, None, term_state)], state_to_idx[(None, None, None, term_state)]] = 1.0

    # ================================================================================================ #
    # [STEP 3] Iterate through "from" states and assign transitions
    # ================================================================================================ #
    # for readability: full states go (inning, half_inning, home_lead, game_state)
    for (half_inning_from,), transition_probs_half in transition_probs.groupby(["inning_topbot"], as_index=False):
        for inning_from, home_lead_from in tqdm(itertools.product(innings, home_leads)):
            for _, row in transition_probs_half.iterrows():
                # extract row info
                game_state_from = row["game_state"]
                game_state_to = row["game_state_post"]
                runs_scored = row["runs"]
                prob = row["prob"]

                # if inning is top half, then runs_scored _detracts_ from the
                # home_lead
                sign = -1 if half_inning_from == "Top" else 1

                home_lead_to = home_lead_from + sign * runs_scored

                # ================================================================================================ #
                # [STEP 3A] Is this an impossible starting spot? If yes, just skip it
                # ================================================================================================ #
                if (inning_from == 1 and half_inning_from == "Top" and home_lead_from > 0) or (
                    inning_from >= 9 and half_inning_from == "Bot" and home_lead_from > 0
                ):
                    continue

                # ================================================================================================ #
                # [STEP 3B] Is the game over by score differential? If so, auto-transition to terminal state
                # ================================================================================================ #
                elif np.abs(home_lead_to) >= MAX_SCORE_DIFFERENTIAL:
                    term_state = "home_loss" if home_lead_to < 0 else "home_win"
                    P_full[
                        state_to_idx[(inning_from, half_inning_from, home_lead_from, game_state_from)],
                        state_to_idx[(None, None, None, term_state)],
                    ] += prob

                # ================================================================================================ #
                # [STEP 3C] Is the game automatically because the home team has scored more in / going into B9+
                # ================================================================================================ #
                # home team WINS
                elif (
                    # walk off
                    (inning_from >= 9 and half_inning_from == "Bot" and home_lead_to > 0)
                    or
                    # shut the door T9
                    (inning_from >= 9 and half_inning_from == "Top" and game_state_to == "---:3" and home_lead_to > 0)
                ):
                    P_full[
                        state_to_idx[(inning_from, half_inning_from, home_lead_from, game_state_from)],
                        state_to_idx[(None, None, None, "home_win")],
                    ] += prob
                # home team LOSES
                elif inning_from >= 9 and half_inning_from == "Bot" and game_state_to == "---:3" and home_lead_to < 0:
                    P_full[
                        state_to_idx[(inning_from, half_inning_from, home_lead_from, game_state_from)],
                        state_to_idx[(None, None, None, "home_loss")],
                    ] += prob

                # ================================================================================================ #
                # [STEP 3D] Is the half inning over?
                # ================================================================================================ #
                elif game_state_to == "---:3":

                    if half_inning_from == "Top":
                        inning_to = inning_from
                        half_inning_to = "Bot"
                    else:
                        inning_to = min(inning_from + 1, 10)
                        half_inning_to = "Top"

                    if inning_to == 10:
                        game_state_to = "-2-:0"
                    else:
                        game_state_to = "---:0"

                    P_full[
                        state_to_idx[(inning_from, half_inning_from, home_lead_from, game_state_from)],
                        state_to_idx[(inning_to, half_inning_to, home_lead_to, game_state_to)],
                    ] += prob

                # ================================================================================================ #
                # [STEP 3E] Otherwise, it's an in-inning advancement. Just update score and game state.
                # ================================================================================================ #
                else:
                    P_full[
                        state_to_idx[(inning_from, half_inning_from, home_lead_from, game_state_from)],
                        state_to_idx[(inning_from, half_inning_from, home_lead_to, game_state_to)],
                    ] += prob
    # ensure row stochasticity on the way out
    assert np.isclose(P_full.sum(axis=1), 1.0).all(), "Row-wise probabilities do not sum to one."
    return P_full, full_states


def calculate_closed_form_win_probs() -> pd.DataFrame:
    """
    Compute home win probabilities in closed form via an absorbing Markov chain.

    Ingests PA-level transition probabilities, expands them into a game-level
    transition matrix, partitions it into transient (Q) and absorbing (R) blocks,
    and solves B = (I - Q)^-1 R directly for the absorption (win/loss) probabilities
    of every transient game state.

    Returns
    -------
    pd.DataFrame
        Absorption probabilities by game state, with columns: inning,
        inning_topbot, home_lead, game_state, home_win_prob, and home_loss_prob.
    """

    # pull in PA-to-PA transition probs
    transition_probs = query_transition_probs()
    LOGGER.info("PA-level transition probabilities ingested.")

    # convert those to a game-wide transition matrix
    P, states = construct_game_transition_matrix(transition_probs)
    LOGGER.info("Game-level transition matrix constructed.")

    # only two terminal states—home win / home loss, so can hard-code.
    Q = P[:-2, :-2]
    R = P[:-2, -2:]

    # make id matrix
    I = np.eye(len(Q))  # noqa: E741

    # solve for the absorbing probabilities directly: B = (I - Q)^-1 * R
    B = np.linalg.solve(I - Q, R)
    LOGGER.info("Absorbing states solved.")

    win_probs = (
        pd.DataFrame(B, columns=["home_win_prob", "home_loss_prob"], index=states[:-2])
        .reset_index()
        .rename(
            columns={"level_0": "inning", "level_1": "inning_topbot", "level_2": "home_lead", "level_3": "game_state"}
        )
    )
    return win_probs


def upload() -> None:
    """
    Calculate closed-form home win probabilities and save them to DuckDB.

    Runs the full closed-form win probability pipeline, selects the reporting
    columns, and writes the result to the strategery win-probability table config.
    """
    win_probs = calculate_closed_form_win_probs()[
        ["game_state", "inning", "inning_topbot", "home_lead", "home_win_prob"]
    ]
    write_parquet(win_probs, "duckdb/table_config/strategery__win_probability.yaml")


if __name__ == "__main__":
    upload()
