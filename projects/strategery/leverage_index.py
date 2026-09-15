import pandas as pd

from baseball.duckdb.database import TABLES, describe_table, write_parquet
from baseball.projects.strategery.constants import GAME_STATES, LEVERAGE_DENOM, MAX_SCORE_DIFFERENTIAL
from baseball.utils.duckdb import query


def calculate_leverage() -> pd.DataFrame:
    """
    Stitches together the ``strategery.re24_transition_probs`` and ``strateger.win_probability``
    tables to calculate leverage index. Leverage denominator is fixed at ``LEVERAGE_DENOM`` for now,
    though this should be recalculated in the future.

    Returns
    -------
    pd.DataFrame
        A dataframe of game states and corresponding leverage index.
    """

    game_state_str = "', '".join(GAME_STATES)
    sql = f"""
    WITH cross_join AS (
        SELECT 
            inning, 
            inning_topbot,
            home_lead,
            game_state
        FROM UNNEST(range(1, 11)) AS t(inning)
        CROSS JOIN UNNEST(['Top', 'Bot']) AS h(inning_topbot)
        CROSS JOIN UNNEST(range(-{MAX_SCORE_DIFFERENTIAL}, {MAX_SCORE_DIFFERENTIAL + 1})) AS l(home_lead)
        CROSS JOIN UNNEST(['{game_state_str}']) AS g(game_state)
    ), 
    pre_post AS (
        SELECT 
            base.inning,
            CASE WHEN tp.game_state_post = '---:3'
                    THEN LEAST(IF(base.inning_topbot = 'Top', base.inning, base.inning + 1), 10)
                ELSE base.inning END AS inning_post,
            -- half inning--
            base.inning_topbot,
            CASE WHEN tp.game_state_post = '---:3' AND base.inning_topbot = 'Top'
                    THEN 'Bot'
                WHEN tp.game_state_post = '---:3' AND base.inning_topbot = 'Bot'
                    THEN 'Top'
                ELSE base.inning_topbot END AS inning_topbot_post,
            -- lead -- 
            base.home_lead,
            base.home_lead + IF(base.inning_topbot = 'Top', -1, 1) * tp.runs AS home_lead_post,
            -- game state --
            base.game_state,
            tp.game_state_post AS game_state_post_transition,
            CASE WHEN tp.game_state_post = '---:3' AND (base.inning <= 8 OR (base.inning_topbot = 'Top' AND base.inning = 9))
                    THEN '---:0'
                WHEN tp.game_state_post = '---:3' AND (base.inning >= 10 OR (base.inning_topbot = 'Bot' AND base.inning = 9))
                    THEN '-2-:0'
                ELSE tp.game_state_post END AS game_state_post,
            tp.runs,
            IF(
                (
                    -- didn't score enough B9+
                    base.inning >= 9 AND 
                    base.inning_topbot = 'Bot' AND 
                    tp.game_state_post = '---:3' AND
                    (base.home_lead + tp.runs < 0)
                ) OR (
                    -- blowout
                    base.inning_topbot = 'Top' AND
                    base.home_lead - tp.runs < -{MAX_SCORE_DIFFERENTIAL}
                ),
                1, 0
            ) AS game_end_home_loss,

            IF(
                (
                    base.inning_topbot = 'Bot' AND
                    base.home_lead + tp.runs > {MAX_SCORE_DIFFERENTIAL}

                ) OR (
                    base.inning >= 9 AND (
                    -- walked it off
                    (base.inning_topbot = 'Bot' AND (base.home_lead + tp.runs > 0))
                    OR
                    -- closed out T9
                    (base.inning_topbot = 'Top' AND tp.game_state_post = '---:3' AND (base.home_lead - tp.runs > 0))
                    )
            ),
            1, 0
            ) AS game_end_home_win,
            tp.n_transitions,
            tp.n_total,
            tp.prob
        FROM cross_join base
        JOIN '{TABLES.strategery.re24_transition_probs}' tp
            USING(inning_topbot, game_state)
    ), 
    pre_post_wp AS (
        SELECT 
            pp.*,
            wp_pre.home_win_prob AS home_win_prob,
            CASE WHEN pp.game_end_home_win = 1 THEN 1.0
                WHEN pp.game_end_home_loss = 1 THEN 0.0
                ELSE wp_post.home_win_prob END AS home_win_prob_post,
        FROM pre_post pp
        JOIN '{TABLES.strategery.win_probability}' wp_pre USING(inning, inning_topbot, home_lead, game_state)
        LEFT JOIN '{TABLES.strategery.win_probability}' wp_post ON 
            pp.inning_post = wp_post.inning AND 
            pp.inning_topbot_post = wp_post.inning_topbot AND
            pp.home_lead_post = wp_post.home_lead AND 
            pp.game_state_post = wp_post.game_state
    ),
    wp_deltas AS (
        SELECT
            game_state,
            inning,
            inning_topbot,
            home_lead,
            SUM(prob) AS prob_check,
            SUM(
                prob * ABS(home_win_prob_post - home_win_prob)
            ) expected_home_win_prob_delta,
        FROM pre_post_wp
        GROUP BY 
            game_state,
            inning,
            inning_topbot,
            home_lead
    )
    SELECT 
        *,
        expected_home_win_prob_delta / {LEVERAGE_DENOM}
    FROM wp_deltas
    """

    lev_df = query(sql)

    return lev_df
