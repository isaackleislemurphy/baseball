import itertools

import pandas as pd
import torch
import torch.distributions as td
from tqdm import tqdm

from baseball.duckdb.tables import TABLES
from baseball.projects.strategery.constants import (
    GAME_STATES,
    RUNS_ARRAY,
)
from baseball.utils.duckdb import query

N_SIMS = 100_000
HOME_LEADS = torch.arange(-10, 11)
PROB_RS_COLS = ["p_" + str(item) for item in RUNS_ARRAY]


def get_re24_values() -> pd.DataFrame:
    """
    Query RE24 table from DuckDB.

    Returns
    -------
    pd.DataFrame
        RE24 values with run probability distributions by game state.
    """
    return query(f"SELECT * FROM '{TABLES.strategery.re24}'")


def make_run_probability_dists(re24: pd.DataFrame) -> dict[tuple[int, str], td.Categorical]:
    """
    Create categorical distributions for runs scored by game state and half inning.

    Parameters
    ----------
    re24 : pd.DataFrame
        RE24 table with run probability columns.

    Returns
    -------
    dict[tuple[int, str], td.Categorical]
        Maps (half_inning, game_state) to categorical distribution over runs.
        half_inning: 0 for top, 1 for bottom.
    """
    # get P(0 runs, 1 runs, ... | game_state, half_inning)
    p_runs = {
        (half_inning, game_state): td.Categorical(
            torch.from_numpy(
                re24.query(f"game_state == '{game_state}' & inning_topbot == '{'Bot' if half_inning else 'Top'}'")[
                    PROB_RS_COLS
                ].values.flatten()
            )
        )
        for (half_inning, game_state) in itertools.product((0, 1), GAME_STATES)
    }
    return p_runs


def calculate_regular_home_win_probs(
    p_runs: dict[tuple[int, str], td.Categorical], hm_win_prob_t10: float
) -> pd.DataFrame:
    """
    Calculate home win probabilities for all regular inning game states via simulation.

    Parameters
    ----------
    p_runs : dict[tuple[int, str], td.Categorical]
        Run probability distributions by (half_inning, game_state).
    hm_win_prob_t10 : float
        Home win probability at start of extras (tied after 9).

    Returns
    -------
    pd.DataFrame
        Win probabilities by game_state, inning, inning_topbot, and home_lead.
        Columns: game_state, inning, inning_topbot, home_lead, home_win_prob.
    """

    # win prob results go here
    wp_results = []
    n_home_leads = len(HOME_LEADS)

    # loop over game state, inning, and half inning (run differentials vectorized)
    for game_state, inning, half_inning in tqdm(itertools.product(GAME_STATES, range(1, 11), (0, 1))):
        # simulate rest of inning runs
        roi_runs = p_runs[(half_inning, game_state)].sample((n_home_leads, N_SIMS))

        # 1.) REGULAR INNINGS
        if inning < 10:
            # how many (regular) innings will the home team bat? Note the min/max trick here to avoid unwieldy
            # if/else statements—if there are no innings left to play, we take a trivial sample of shape
            # (1, n_home_leads, N_SIMS) so that `.sample()` doesn't break, but we promptly zero it out with
            # `min(hm_inn_rog, 1)`. In-line way of creating zeros.
            hm_inn_rog = 10 - inning - half_inning
            hm_runs = p_runs[(1, "---:0")].sample((max(hm_inn_rog, 1), n_home_leads, N_SIMS)).sum(axis=0) * min(
                hm_inn_rog, 1
            )

            # how many (regular) innings will the opposing team bat (excl. this one)
            rd_inn_rog = 9 - inning
            rd_runs = p_runs[(0, "---:0")].sample((max(rd_inn_rog, 1), n_home_leads, N_SIMS)).sum(axis=0) * min(
                rd_inn_rog, 1
            )
            # tally combine roi + rog runs
            if half_inning == 0:
                rd_runs += roi_runs
            else:
                hm_runs += roi_runs

        # 2.) EXTRA INNINGS
        else:
            # figure out how much home team scored; use ROI if bottom-half, otherwise draw from zombie-runner
            hm_runs = roi_runs if half_inning else p_runs[(1, "-2-:0")].sample((n_home_leads, N_SIMS))
            # if it's bottom, road team scored "0" for the inning that we've already conditioned on;
            # if it's top, use the roi runs
            rd_runs = torch.zeros_like(hm_runs) if half_inning else roi_runs

        # add on (vectorized) home leads
        hm_runs += HOME_LEADS[:, None]

        # calculate wins. TODO: recursive extras computation
        p_home_win = (hm_runs > rd_runs).to(torch.float) + hm_win_prob_t10 * (hm_runs == rd_runs).to(torch.float)
        # average over win probs.
        p_home_win = p_home_win.mean(axis=1)

        wp_results += [
            pd.DataFrame(
                {
                    "game_state": [game_state] * n_home_leads,
                    "inning": [inning] * n_home_leads,
                    "inning_topbot": ["Top" if half_inning == 0 else "Bot"] * n_home_leads,
                    "home_lead": HOME_LEADS.numpy(),
                    "home_win_prob": p_home_win.numpy(),
                }
            )
        ]
    wp_results = pd.concat(wp_results, axis=0)

    return wp_results


def calculate_start_of_extras_home_win_probs(p_runs: dict[tuple[int, str], td.Categorical]) -> float:
    """
    Calculate home win probability at start of 10th inning (tied after 9).

    Uses geometric series to account for infinite extra innings with runner on 2nd.

    Parameters
    ----------
    p_runs : dict[tuple[int, str], td.Categorical]
        Run probability distributions by (half_inning, game_state).

    Returns
    -------
    float
        Home team win probability when entering extras tied.
    """
    # sample T10 runs scored
    rd_roi_runs = p_runs[(0, "-2-:0")].sample((N_SIMS,))
    # sample B10 runs scored
    hm_roi_runs = p_runs[(1, "-2-:0")].sample((N_SIMS,))

    # probability home team walks it of B10
    p_home_win_this_inn = (hm_roi_runs > rd_roi_runs).to(torch.float).mean().item()
    # probability we play 11
    p_another_inn = (hm_roi_runs == rd_roi_runs).to(torch.float).mean().item()

    # solve for geometric series: p(w10) + p(w11) * p(t10) + p(w12) * p(t11) * p(t10) + ...
    # assuming win probs hold from 10+, which probably isn't true but oh well
    return p_home_win_this_inn / (1 - p_another_inn)


if __name__ == "__main__":
    re24 = get_re24_values()
    p_runs = make_run_probability_dists(re24)
    hm_win_prob_t10 = calculate_start_of_extras_home_win_probs(p_runs)
    wp_results = calculate_regular_home_win_probs(p_runs, hm_win_prob_t10)
