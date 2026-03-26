"""
Hierarchical fastball shape model using PyMC.

This module fits a hierarchical Bayesian model for fastball pitch shape
(release speed, release height, horizontal break, vertical break) with
separate treatment for four-seam fastballs and sinkers.

The model estimates:
- Global (league-wide) baselines by pitch type
- Pitcher-level mean fastball shape
- Game-level deviations around each pitcher's baseline

Two model variants are supported:
1. A diagonal-covariance version used for computational and memory efficiency.
2. A full multivariate normal (MVN) version with correlated features, which
   is statistically more appropriate but substantially more expensive.

The diagonal model and the use of variational inference are (i) deliberate
engineering compromises to allow fitting on a local laptop and (ii) more defensible
on the grounds that only the posterior means -- plain old point estimates -- will be
fed into the pitch quality model. These approximations/shortcuts are not claimed to be
theoretically optimal, only practical. And if I ever end up trying to take advantage of the
full probibilistic distribution, rather than mere point estimates, I know I'll have to find
a way to sample. Nonetheless, I left in toggles so that if you ever decide to run this code
with more juice, you'll be able to do the more rigorous model and estimation.

To that end, I'm also fitting the models within season, when clearly the more
rigorous thing would be to add season as a hierarchy. Again, purely a memory play...
if anyone is brave enough to take this for a spin with more memory, would be a no-brainer
to cast down a seasonal index and have a truly hierarchical (player, season, game) model.
"""

import argparse
import os
from typing import Literal

import arviz as az

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from xarray import DataArray

from baseball.projects.pitch_quality.data.etl import load_pitch_quality_model_data
from baseball.utils.general import str2bool

print(f"Running on PyMC v{pm.__version__}")

FB_DIFF_COLS = ["release_speed", "release_pos_z", "pfx_x", "pfx_z", "arm_angle"]
DIM = len(FB_DIFF_COLS)
FA_TYPES = ("FF", "SI")
KEY_COLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]

# filepath crap
DIR_PATH = os.environ.get("PYTHONPATH")
FASTBALL_DIFF_PATH = os.path.join(DIR_PATH, "baseball", "duckdb", "model_outputs", "smoothed_fastball_shapes")
PLAYER_MEANS_PATH = os.path.join(FASTBALL_DIFF_PATH, "player_season", "smoothed_player_means_{season}.parquet")
PLAYER_GAME_MEANS_PATH = os.path.join(
    FASTBALL_DIFF_PATH, "player_season_game", "smoothed_player_game_means_{season}.parquet"
)

# test pitchers: nola, wheeler, kerkering, hoff, zeus, strahm, banks, skenes, fairbanks
TEST_SUBSET_PITCHERS = (605400, 554430, 689147, 656046, 666200, 621381, 621383, 694973, 664126)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for fastball shape model estimation.

    This helper defines and parses the minimal set of CLI arguments required
    to run the fastball shape model from the command line.

    Arguments
    ---------
    --season : int
        MLB season to fit the model on (e.g., 2023). Used to subset the pitch
        data before model construction.
    --model_type : {"diagonal", "mvn"}
        Specifies the covariance structure of the hierarchical model.
        - "diagonal": Diagonal covariance approximation. Faster and more
          memory-efficient.
        - "mvn": Full multivariate normal model with correlated features.
          Statistically preferable but takes longer, particularly
          when sampling.
    --use_test_subset : bool
        If True, restricts fit/sampling to a smal subset of pitchers. Helpful when tinkering
        with the model or debugging. The default is false.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with attributes `season` and `model_type`.
    """
    parser = argparse.ArgumentParser(description="Estimate the fastball shape posterior for a given season")

    parser.add_argument("--season", type=int, required=True, help="Season for fitting (e.g. 2023)")

    parser.add_argument(
        "--model_type",
        type=str,
        default="diagonal",
        choices=["diagonal", "mvn"],
        help="Which model to run: 'diagonal' or 'mvn`",
    )

    parser.add_argument(
        "--use_test_subset",
        type=str2bool,
        default=False,
        help="If True, will restrict fit/sampling to a small subset of pitchers. Probably False.",
    )

    return parser.parse_args()


def load_fastball_data(season: int, use_test_subset: bool = False) -> pd.DataFrame:
    """
    Load and minimally clean fastball pitch-level data.

    Parameters
    ----------
    season : int
        Season on which to fit data. Again, this should be hierarchically treated,
        but modeling within season to save on memory. See note above.
    use_test_subset : bool, default=False
        If True, whittles dataset down to the predetermined pitchers in `TEST_SUBSET`.
        You should ever only use this when iterating/debugging.

    Returns
    -------
    pd.DataFrame
        Pitch-level fastball data with:
        - Only FF and SI pitch types
        - A numeric fastball-type index (`fa_idx`)
        - Rows with missing fastball feature values removed
    """

    # pull in all the pitches
    pitch_data_df = load_pitch_quality_model_data(date_min=f"{season}-01-01", date_max=f"{season}-12-31")

    if use_test_subset:
        print("WARNING—test subset in use for smoothed FA shapes")
        pitch_data_df = pitch_data_df.query(f"pitcher in {TEST_SUBSET_PITCHERS}")

    # whittle it down to fastballs
    pitch_data_df = pitch_data_df.loc[pitch_data_df["pitch_type"].isin(FA_TYPES)].reset_index(drop=True)

    # index fastballs
    pitch_data_df["fa_idx"] = pitch_data_df["pitch_type"].replace({item: i for i, item in enumerate(FA_TYPES)})

    # drop missing-valued columns
    pitch_data_df = pitch_data_df.dropna(subset=FB_DIFF_COLS).reset_index(drop=True)

    return pitch_data_df


def index_games_by_pitcher(pitch_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a per-pitcher, per-season game index.

    Games are indexed sequentially for each pitcher-season combination.
    This index is used to model game-to-game deviations around a pitcher's
    baseline fastball shape.

    Parameters
    ----------
    pitch_data_df : pd.DataFrame
        Pitch-level data containing pitcher and game identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame mapping each (pitcher, season, game) to an integer `game_idx`.
    """
    # sort unique games by pitcher, in order to index them
    game_index_df = (
        pitch_data_df.groupby(["pitcher", "game_date", "game_pk"], as_index=False)
        .head(1)
        .sort_values(KEY_COLS)[["pitcher", "season", "game_date", "game_pk"]]
        .assign(game_idx=1)
    )
    # index the games
    game_index_df["game_idx"] = game_index_df.groupby(["pitcher", "season"])["game_idx"].cumcount()

    return game_index_df


def load_and_process_model_data(season: int, **kwargs: dict) -> pd.DataFrame:
    """
    Load, clean, and index data for hierarchical fastball modeling.

    This function prepares all data structures required by the PyMC model:
    - Pitch-level observations
    - Pitcher indices
    - Game indices
    - Boolean masks for pitch type routing (FF vs SI)

    Parameters
    ----------
    season : int
        Season on which to fit data. Again, this should be hierarchically treated,
        but modeling within season to save on memory. See note above.

    **kwargs: dict
        Keyword args for `load_fastball_data()`

    Returns
    -------
    dict
        Dictionary containing:
        - `pitch_data_df`: cleaned pitch-level data
        - `game_index_df`: game index mapping
        - `p_idx`: pitcher index per pitch
        - `pitchers`: unique pitcher IDs
        - `P`: number of pitchers
        - `g_idx`: game index per pitch
        - `G`: number of games
        - `mask_ff`: boolean mask for four-seam fastballs
        - `mask_si`: boolean mask for sinkers
    """
    # pull in the pitches
    pitch_data_df = load_fastball_data(season=season, **kwargs)

    # index the games
    game_index_df = index_games_by_pitcher(pitch_data_df)

    # drop extraneous columns. Helps with memory, and general clutter.
    pitch_data_df = pitch_data_df[KEY_COLS + ["pitch_type", "fa_idx"] + FB_DIFF_COLS].merge(
        game_index_df, on=["pitcher", "game_date", "game_pk"]
    )

    # stash the data
    data = dict(
        game_index_df=game_index_df,
        pitch_data_df=pitch_data_df,
    )

    # index the pitchers and save them to `data`
    data["p_idx"], data["pitchers"] = pitch_data_df["pitcher"].factorize()
    data["P"] = data["pitchers"].size

    # add game index to data
    data["g_idx"], data["G"] = pitch_data_df["game_idx"].values, pitch_data_df["game_idx"].max() + 1

    # mask for fastball type
    data["mask_ff"] = pitch_data_df["pitch_type"].values == "FF"
    data["mask_si"] = pitch_data_df["pitch_type"].values == "SI"

    return data


def _make_lkj_cholesky(dim: int, suffix: str = "", eta: float = 4.0) -> pt.TensorVariable:
    """
    Construct an LKJ-based Cholesky factor for a covariance matrix.

    This helper builds a Cholesky factor using an LKJ prior on the correlation
    matrix and exponential priors on marginal standard deviations. It is used
    in the full MVN version of the model to allow correlated pitch features.

    Parameters
    ----------
    dim : int
        Dimensionality of the covariance matrix.
    suffix : str, optional
        Suffix appended to PyMC variable names to keep them distinct.
    eta : float, default 4.0
        LKJ shape parameter controlling shrinkage toward zero correlation.

    Returns
    -------
    pt.TensorVariable
        Cholesky factor of the covariance matrix.
    """
    # mu_p ~ MVN(0, Sigma_p)
    L, corr, _ = pm.LKJCholeskyCov("L" + suffix, eta=eta, n=dim, sd_dist=pm.Exponential.dist(1.0, size=dim))
    pm.Deterministic("corr" + suffix, corr)

    return L


def instantiate_model(data: dict, model_type: Literal["diagonal", "mvn"] = "diagonal") -> pm.Model:
    """
    Instantiate a hierarchical PyMC model for fastball shape.

    The model captures variation at three levels:
    - League-wide baselines by pitch type (FF vs SI)
    - Pitcher-level mean fastball shape
    - Game-level deviations around each pitcher's baseline

    Two covariance structures are supported:
    - `diagonal`: assumes independence across pitch features. This is a
    computational shortcut chosen to reduce memory use and runtime when
    fitting locally.
    - `mvn`: uses full multivariate normals with correlated features. This
    formulation is statistically more appropriate but significantly more
    expensive to sample.

    The diagonal model and the use of variational inference are pragmatic
    engineering choices, not claims of optimal Bayesian practice.

    Parameters
    ----------
    data : dict
        Processed pitch-level data dictionary returned by
        `load_and_process_model_data`.
    model_type : {"diagonal", "mvn"}, default "diagonal"
        Choice of covariance structure.

    Returns
    -------
    pm.Model
        An instantiated (but unsampled) PyMC model.
    """

    # unpack the data
    p_idx, _, P = data["p_idx"], data["pitchers"], data["P"]
    g_idx, G = data["g_idx"], data["G"]
    mask_ff, mask_si = data["mask_ff"], data["mask_si"]
    pitch_data_df = data["pitch_data_df"]

    with pm.Model() as model:

        # ------------------------------------------------------------
        # Data / preprocessing inside the model
        # ------------------------------------------------------------

        # Observed pitch features: shape (N_pitches, DIM)
        Y = pitch_data_df[FB_DIFF_COLS].values

        # Global (league-wide) mean and std for standardization
        # These are fixed constants, not random variables
        mu_Y = pitch_data_df[FB_DIFF_COLS].mean().values[None, :]
        sigma_Y = pitch_data_df[FB_DIFF_COLS].std().values[None, :]

        # Standardized observations (roughly mean 0, variance 1)
        Y_ = (Y - mu_Y) / sigma_Y
        # save the scaling for inverse transforms
        pm.ConstantData("mu_Y", mu_Y)
        pm.ConstantData("sigma_Y", sigma_Y)

        # ------------------------------------------------------------
        # Global intercepts by pitch type
        # ------------------------------------------------------------

        # alpha contains separate intercept vectors for FF and SI:
        #   alpha[:, :DIM]  -> FF baseline
        #   alpha[:, DIM:]  -> SI baseline
        #
        # Since Y_ is standardized, these represent global deviations
        # from league-average pitch shape by pitch type.
        alpha = pm.Normal("alpha", 0, 1, size=(1, DIM * 2))

        # ============================================================
        # DIAGONAL COVARIANCE VERSION (computational shortcut)
        # ============================================================
        if model_type == "diagonal":

            # --------------------------------------------------------
            # Player-level latent means
            # --------------------------------------------------------

            # Feature-wise scale of between-pitcher variation.
            # Fat tails for the weirdos.
            sigma_p = pm.HalfCauchy("sigma_p", 5.0, shape=(1, DIM * 2))

            # mu_p[p, :] is the average fastball shape for pitcher p
            # Separate FF and SI blocks are concatenated (not strictly necessary here, but
            # matches formulation of the MVN setup below)
            mu_p = pm.Normal("mu_p", 0, sigma_p, shape=(P, DIM * 2))

            # --------------------------------------------------------
            # Player–game deviations (within-pitcher variation)
            # --------------------------------------------------------

            # Game-over-game variation for each feature
            sigma_z_pg = pm.Exponential("sigma_pg", 1, shape=(1, 1, DIM * 2), initval=1e-2 * np.ones((1, 1, DIM * 2)))

            # eta_pg[p, g, :] is how pitcher p deviates in game g from their
            # baseline in mu_p[p, :]
            z_pg = pm.Normal("z_pg", 0, sigma_z_pg, shape=(P, G, DIM * 2))
            eta_pg = z_pg.cumsum(axis=1)

            # --------------------------------------------------------
            # Likelihood: four-seam fastballs
            # --------------------------------------------------------

            # Observation-level noise for FFs (pitch-to-pitch variation)
            sigma_ff = pm.HalfNormal("sigma_ff", 1, shape=(1, DIM))

            # Expected value is:
            #   global FF baseline
            # + pitcher FF mean
            # + pitcher-game FF deviation
            _ = pm.Normal(
                "lkhd_ff",
                alpha[:, :DIM] + mu_p[p_idx[mask_ff], :DIM] + eta_pg[p_idx[mask_ff], g_idx[mask_ff], :DIM],
                sigma_ff,
                observed=Y_[mask_ff],
            )

            # --------------------------------------------------------
            # Likelihood: sinkers
            # --------------------------------------------------------

            # Separate observation noise for sinkers
            sigma_si = pm.HalfNormal("sigma_si", 1, shape=(1, DIM))

            # Same structure as FF, but using SI block
            _ = pm.Normal(
                "lkhd_si",
                alpha[:, DIM:] + mu_p[p_idx[mask_si], DIM:] + eta_pg[p_idx[mask_si], g_idx[mask_si], DIM:],
                sigma_si,
                observed=Y_[mask_si],
            )

        # ============================================================
        # FULL MVN VERSION (statistically more correct, expensive)
        # ============================================================
        else:

            # --------------------------------------------------------
            # Player-level means: mu_p ~ MVN(0, Sigma_p)
            # --------------------------------------------------------

            # Cholesky factor of player covariance matrix to allow
            # correlation across pitch features. Note the non-centered
            # parameterization here.

            L_p = _make_lkj_cholesky(dim=DIM * 2, eta=4.0, suffix="_p")
            Z_p = pm.Normal("z_p", 0, 1, size=(P, DIM * 2))
            mu_p = pm.Deterministic("mu_p", pm.math.dot(L_p, Z_p.T).T)

            # --------------------------------------------------------
            # Player–game effects: eta_pg ~ MVN(0, Sigma_pg)
            # --------------------------------------------------------

            # Cholesky factor for game-level covariance. Again, non-
            # centered parameterization
            L_pg = _make_lkj_cholesky(dim=DIM * 2, eta=4.0, suffix="_pg")
            Z_pg = pm.Normal("z_pg", 0, 1, size=(P, G, DIM * 2))
            eta_pg = pm.math.dot(Z_pg, L_pg.T)

            # --------------------------------------------------------
            # MVN likelihood: four-seam fastballs
            # --------------------------------------------------------

            # Observation-level covariance for FFs
            L_ff = _make_lkj_cholesky(dim=DIM, eta=4.0, suffix="_ff")

            _ = pm.MvNormal(
                "lkhd_ff",
                alpha[:, :DIM] + mu_p[p_idx[mask_ff], :DIM] + eta_pg[p_idx[mask_ff], g_idx[mask_ff], :DIM],
                chol=L_ff,
                observed=Y_[mask_ff],
            )

            # --------------------------------------------------------
            # MVN likelihood: sinkers
            # --------------------------------------------------------

            # Observation-level covariance for sinkers
            L_si = _make_lkj_cholesky(dim=DIM, eta=4.0, suffix="_si")

            _ = pm.MvNormal(
                "lkhd_si",
                alpha[:, DIM:] + mu_p[p_idx[mask_si], DIM:] + eta_pg[p_idx[mask_si], g_idx[mask_si], DIM:],
                chol=L_si,
                observed=Y_[mask_si],
            )
    return model


def _add_suffix(x: list[str], suffix) -> list:
    """Adds a suffix to a list of strings; quick helper for column munging"""
    return [item + suffix for item in x]


def sample_posterior(
    model: pm.Model, nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "nutpie", **kwargs: dict
) -> az.InferenceData:
    """
    Sample the posterior distribution using NUTS.

    This function runs full MCMC sampling and should generally only be used
    with the MVN model or on smaller datasets due to its computational cost.

    Parameters
    ----------
    model : pm.Model
        Instantiated PyMC model.
    nuts_sampler : {"pymc", "nutpie", "numpyro", "blackjax"}, default "nutpie"
        Backend NUTS sampler to use.
    **kwargs
        Passed directly to `pm.sample`.

    Returns
    -------
    arviz.InferenceData
        Posterior samples.
    """
    with model:
        trace = pm.sample(nuts_sampler=nuts_sampler, **kwargs)
    return trace


def approximate_posterior(
    model: pm.Model,
    approximation_method: Literal["dadvi", "advi", "fullrank_advi", "svgd", "asvgd"] = "advi",
    n_samples: int = 1_000,
    **kwargs: dict,
) -> tuple[pm.variational.approximations.MeanField, az.InferenceData]:
    """
    Approximate the posterior using variational inference.

    This is primarily intended for the diagonal-covariance model, where full
    MCMC sampling would be unnecessarily slow or memory-intensive on a local
    machine. Variational inference trades some fidelity for speed and
    tractability.

    Parameters
    ----------
    model : pm.Model
        Instantiated PyMC model.
    approximation_method : {"dadvi", "advi", "fullrank_advi", "svgd", "asvgd"}, default "advi"
        Variational inference method.
    n_samples : int, default 1000
        Number of posterior samples to draw from the approximation.
    **kwargs
        Passed to the underlying PyMC fitting routine.

    Returns
    -------
    tuple
        - Variational approximation object
        - InferenceData containing approximate posterior samples
    """
    if approximation_method == "dadvi":
        from pymc_extras.inference import fit

        _fit_fn = fit
    else:
        _fit_fn = pm.fit

    with model:
        approx = _fit_fn(method=approximation_method, **kwargs)
        trace = approx.sample(n_samples)
    return approx, trace


def _extract_samples(x: DataArray) -> np.ndarray:
    """
    Extracts and stacks samples from a PyMC trace across chains.

    Parameters
    ----------
    x : DataArray
        Input data array containing MCMC samples, typically with
        dimensions (chains, draws, ...).

    Returns
    -------
    numpy.ndarray
        Stacked array with chains concatenated along the first axis,
        resulting in shape (chains * draws, ...).
    """
    return np.concatenate(x.values)


def extract_posterior_means(trace: az.InferenceData, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract posterior mean estimates for pitcher and pitcher-game effects.

    This function converts standardized latent variables back to the original
    feature scale and produces tidy DataFrames of:
    - Pitcher-level seasonal fastball shape
    - Pitcher-game-level fastball shape deviations

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples (actually sampled or approximate).
    data : dict
        Data dictionary produced by `load_and_process_model_data`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Pitcher-level posterior means
        - Pitcher-game-level posterior means
    """
    # extract posterior samples of seasonal player means:
    # (draws, 1, DIM * 2) + (draws, P, DIM * 2) --> (draws, P, DIM * 2)
    mu_p_samples = _extract_samples(trace.posterior.alpha) + _extract_samples(trace.posterior.mu_p)

    # extract posterior samples of player game means:
    # (draws, P, 1, DIM * 2) + (draws, P, G, DIM * 2)--> (draws, P, G, DIM * 2)
    eta_pg_samples = mu_p_samples[:, :, None, :] + _extract_samples(trace.posterior.z_pg).cumsum(axis=2)

    # extract scaling parameters, to unscale
    sigma_y = trace.constant_data.sigma_Y.values.repeat(2, 0).flatten()[None, :]
    mu_y = trace.constant_data.mu_Y.values.repeat(2, 0).flatten()[None, :]

    # get pitchers
    pitchers = data["pitchers"]

    # get widened column names
    _wide_cols = _add_suffix(FB_DIFF_COLS, "_FF") + _add_suffix(FB_DIFF_COLS, "_SI")

    # dataframe of seasonal player posterior means...return this.
    smoothed_player_means = pd.DataFrame(mu_p_samples.mean(0) * sigma_y + mu_y, columns=_wide_cols).assign(
        pitcher=pitchers
    )[["pitcher"] + _wide_cols]
    smoothed_player_game_means = eta_pg_samples.mean(0)

    # dataframe of player-game posterior means...return this
    smoothed_player_game_means = pd.concat(
        [
            pd.DataFrame(item * sigma_y + mu_y, columns=_wide_cols)
            .assign(pitcher=pitchers[i], game_idx=np.arange(item.shape[0]))
            .merge(data["game_index_df"], on=["pitcher", "game_idx"])
            for i, item in enumerate(smoothed_player_game_means)
        ],
        axis=0,
    ).reset_index(drop=True)
    smoothed_player_game_means = smoothed_player_game_means[
        [item for item in smoothed_player_game_means.columns if item not in _wide_cols] + _wide_cols
    ]

    return smoothed_player_means, smoothed_player_game_means


def main() -> None:
    """Main function"""
    # args
    args = parse_args()
    season, model_type, use_test_subset = args.season, args.model_type, args.use_test_subset

    # pull in data
    data = load_and_process_model_data(season=season, use_test_subset=use_test_subset)
    print("Player-game FA shape data loaded")

    # set up model
    model = instantiate_model(data, model_type=model_type)
    print("Player-game FA shape model instantiated")

    # approximate
    approx, trace = approximate_posterior(model, n=50_000)
    print("Player-game FA shape posterior approximated.")

    # TODO: diagnostics in here

    # extract the posterior means
    smoothed_player_means, smoothed_player_game_means = extract_posterior_means(trace, data)
    smoothed_player_means["season"] = season  # add in a season column to player means
    print("Player-game FA shape posterior means extracted")

    # save the game means (a)
    smoothed_player_means.to_parquet(PLAYER_MEANS_PATH.format(season=season), index=False)
    smoothed_player_game_means.to_parquet(PLAYER_GAME_MEANS_PATH.format(season=season), index=False)
    print(f"Game-by-game FA shapes for {season} saved.")


if __name__ == "__main__":
    main()
