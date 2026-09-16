""""""

import itertools

import numpy as np
import pandas as pd

from baseball.duckdb.database import TABLES
from baseball.projects.strategery.win_probability import construct_game_transition_matrix, query_transition_probs
from baseball.utils.duckdb import query
from baseball.utils.logging import get_logger

LOGGER = get_logger()


def query_leverage() -> pd.DataFrame:
    """
    Load the leverage index table from DuckDB.

    Returns
    -------
    pd.DataFrame
        Leverage indices keyed by inning, inning_topbot, home_lead, and game_state,
        with a leverage_index column.
    """

    sql = f"""
    SELECT
        inning,
        inning_topbot,
        home_lead,
        game_state,
        leverage_index
    FROM '{TABLES.strategery.leverage_index}'
    """
    leverage = query(sql)

    return leverage


def calculate_expected_rest_of_game_visits() -> tuple[np.ndarray, list[tuple]]:
    """
    Compute the expected number of rest-of-game visits to each game state.

    Builds the game-level absorbing Markov transition matrix, extracts the transient
    block Q, and solves the fundamental matrix V = (I - Q)^-1. Entry V[i, j] gives the
    expected number of times state j is visited before absorption when starting from
    state i.

    Returns
    -------
    V : np.ndarray
        The fundamental matrix (I - Q)^-1 over the transient (non-terminal) states.
        Shape is (n_transient, n_transient).
    states : list[tuple]
        The ordered list of all states indexing the full transition matrix, including
        the two terminal states at the end. Each state is a tuple of
        (inning, inning_topbot, home_lead, game_state).
    """
    # pull in PA-to-PA transition probs
    transition_probs = query_transition_probs()
    LOGGER.info("PA-level transition probabilities ingested.")

    # convert those to a game-wide transition matrix
    P, states = construct_game_transition_matrix(transition_probs)
    state_to_idx = {state: i for i, state in enumerate(states)}
    LOGGER.info("Game-level transition matrix constructed.")

    Q = P[:-2, :-2]
    # make id matrix
    I = np.eye(len(Q))  # noqa: E741

    V = np.linalg.solve(I - Q, I)
    LOGGER.info("Expected rest-of-game visits calculated.")

    return V, states


def calculate_expected_rest_of_game_leverage_visits(
    inclusive: bool = True, leverage_thresholds: tuple[int, ...] = (1, 2, 3, 4)
) -> pd.DataFrame:
    """
    Compute expected rest-of-game visits to high-leverage states by threshold.

    For each starting game state, uses the fundamental matrix of expected visits to
    count the expected number of remaining plate appearances spent in states whose
    leverage index meets or exceeds each provided threshold.

    Parameters
    ----------
    inclusive : bool, optional
        If True, count the current (starting) game state toward the total above each
        threshold, by default True.
    leverage_thresholds : tuple[int, ...], optional
        Leverage index thresholds at which to tally expected visits, by default
        (1, 2, 3, 4). One output column is produced per threshold.

    Returns
    -------
    pd.DataFrame
        Expected high-leverage visit counts keyed by inning, inning_topbot, home_lead,
        and game_state, with one `expected_visits_<lt>_plus` column per threshold.
    """
    # calculate expected visits
    V, states = calculate_expected_rest_of_game_visits()

    # pull in leverage indices
    leverage = query_leverage()
    # maps (inning, inning_topbot, home_lead, game_state) --> LI
    leverage_map = {
        (row["inning"], row["inning_topbot"], row["home_lead"], row["game_state"]): row["leverage_index"]
        for _, row in leverage.iterrows()
    }

    # (1, |states| - 2) matrix of the leverage index at every state you could visit in the future.
    # NOTE—this is row-aligned with states[:-2], so it doubles as each starting state's own LI.
    L = np.array([leverage_map[state] for state in states[:-2]])[None, :]

    # column names, build them up front
    visit_cols = [f"expected_visits_{lt}_plus" for lt in leverage_thresholds]

    # from each starting state, count the expected number of rest-of-game visits to a
    # leverage index at/above each threshold. if inclusive, tack on the starting state
    # itself (its LI lives on the diagonal of L, positionally, so just add (L >= lt)).
    expected_rog_visits = np.hstack(
        [
            np.sum(V * (L >= lt).astype(float), axis=1, keepdims=True) + (inclusive * (L.T >= lt).astype(float))
            for lt in leverage_thresholds
        ]
    )

    # convert to dataframe + add in state as pri-key
    expected_rog_visits = pd.concat(
        [
            pd.DataFrame(states[:-2], columns=["inning", "inning_topbot", "home_lead", "game_state"]),
            pd.DataFrame(expected_rog_visits, columns=visit_cols),
        ],
        axis=1,
    )

    return expected_rog_visits
