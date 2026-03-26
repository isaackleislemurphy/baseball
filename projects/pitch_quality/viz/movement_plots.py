import argparse
import itertools
from datetime import datetime, timedelta

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from baseball.data.chadwick.ids import get_bam_id_from_name
from baseball.data.savant.pitch.load import load_pitch_data

WHITE = "#FFFFFF"
PITCH_COLORS = {
    "FF": "#FB2C36",  # red
    "SI": "#FF8904",  # orange
    "CH": "#31C950",  # green
    "SL": "#FFDF20",  # yellow
    "ST": "#B7950B",  # gold
    "FC": "#873600",  # brown
    # splitter family
    "FS": "#00A390",
    "SC": "#00A390",
    "FO": "#00A390",  # turqoise
    # curve family
    "CU": "#21BCFF",
    "KC": "#21BCFF",
    "SV": "#21BCFF",  # carolina
}

PITCH_CMAPS = {
    pitch_type: LinearSegmentedColormap.from_list(f"custom_{color}", [WHITE, color])
    # sns.color_palette(["#ffffff", color], as_cmap=True)
    for pitch_type, color in PITCH_COLORS.items()
}


def load_and_process_pitch_mvmt_data(pitcher: str, date_min: str, date_max: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches and processes pitch movement data for a specific pitcher.

    Retrieves the MLBAM ID for the given pitcher name, loads their pitch data
    within the specified date range, calculates mean movement profiles for each
    pitch type, and sorts the data by pitch usage frequency.

    Parameters
    ----------
    pitcher : str
        The full name of the pitcher (e.g., "Gerrit Cole").
    date_min : str
        The start date for the data query in 'YYYY-MM-DD' format.
    date_max : str
        The end date for the data query in 'YYYY-MM-DD' format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - pitch_df_pitcher: Raw pitch data, sorted so the most frequently thrown
          pitches appear last (useful for z-order plotting).
        - pitch_df_pitcher_means: Aggregated data containing the mean horizontal
          movement, vertical movement, arm angle, release speed, and pitch count
          for each pitch type.
    """
    # convert to BAM ID
    pitcher_id = get_bam_id_from_name(pitcher)

    # load in pitches over the relevant timeframe, and filter down to the pitcher of interest
    pitch_df_pitcher = load_pitch_data(
        date_min=date_min, date_max=date_max, addl_where_clause=f"AND p.pitcher = {pitcher_id}"
    )

    # get the mean profile for the pitch, as an extra item to plot
    pitch_df_pitcher_means = (
        pitch_df_pitcher.assign(n=1)
        .groupby(["pitch_type", "throws"], as_index=False)
        .agg({item: "mean" for item in ("pfx_x", "pfx_z", "arm_angle", "release_speed")} | {"n": "sum"})
        .sort_values("n")
        .reset_index(drop=True)
    )
    # sort `pitch_df_pitcher` by number of pitches thrown, so pitches thrown more are on "top" of the plot.
    pitch_df_pitcher = (
        pitch_df_pitcher.merge(pitch_df_pitcher_means[["pitch_type", "throws", "n"]], on=["pitch_type", "throws"])
        .sort_values("n")
        .reset_index(drop=True)
    )

    return pitch_df_pitcher, pitch_df_pitcher_means


def _plot_pitch_movement_data(
    fig: Figure, ax: Axes, pitch_df_pitcher: pd.DataFrame, pitch_df_pitcher_means: pd.DataFrame
) -> tuple[Figure, Axes]:
    """
    Plots the core pitch movement data including KDE contours, arm angles, and mean points.

    Iterates through the pitch data from least used to most used (managed via zorder)
    to plot density contours, draws a dashed line representing the average arm angle
    adjusted for handedness, and overlays scatter points representing the mean movement
    scaled by pitch usage percentage.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The current Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The current Matplotlib axes object to draw the data on.
    pitch_df_pitcher : pd.DataFrame
        Raw pitch data sorted by frequency, containing movement and pitch type data.
    pitch_df_pitcher_means : pd.DataFrame
        Aggregated data containing the mean movement and metrics for each pitch type.

    Returns
    -------
    tuple[Figure, Axes]
        The updated Matplotlib figure and axes objects containing the plotted data.
    """

    sns.set_style("white")

    # use this to configure what's furthest back/forwards
    zorder = 0

    # plot the movement contours. Note that by including number of pitches `n` in the grouping, we trick the
    # `.groupby()` into iterating smallest sample --> largest sample
    for (n, pitch_type), df in pitch_df_pitcher.groupby(["n", "pitch_type"], as_index=False):
        sns.kdeplot(
            x=-df.pfx_x, y=df.pfx_z, cmap=PITCH_CMAPS[pitch_type], fill=True, ax=ax, alpha=0.75, levels=3, zorder=zorder
        )
        sns.kdeplot(
            x=-df.pfx_x,
            y=df.pfx_z,
            color=PITCH_COLORS[pitch_type],
            fill=False,
            ax=ax,
            levels=3,
            zorder=zorder,
            alpha=0.25,
        )
        zorder += 1

    # plot the arm angles
    for (n, pitch_type, throws), df in pitch_df_pitcher_means.groupby(["n", "pitch_type", "throws"], as_index=False):
        # bump angle to radians
        arm_angle_rad = np.deg2rad(df.arm_angle.item())
        # flip cosine if LHP
        throws_sign = -1 if throws == "L" else 1
        # plot the arm angle
        ax.plot(
            [0, throws_sign * 24 * np.cos(arm_angle_rad)],
            [0, 24 * np.sin(arm_angle_rad)],
            color=PITCH_COLORS[pitch_type],
            linestyle="dotted",
            zorder=zorder,
        )
        zorder += 1

    # calculate usage
    pitch_df_pitcher_means["pct"] = pitch_df_pitcher_means["n"] / pitch_df_pitcher_means["n"].sum()

    # plot the movement means
    for (n, pitch_type), df in pitch_df_pitcher_means.groupby(["n", "pitch_type"], as_index=False):
        label = f"{pitch_type}: {df.release_speed.round(1).item()} ({n})"
        ax.scatter(
            x=-df.pfx_x,
            y=df.pfx_z,
            color=PITCH_COLORS[pitch_type],
            edgecolor="black",
            zorder=zorder,
            s=df.pct.item() * 500,
            label=label,
        )
    return fig, ax


def _plot_pitch_movement_aesthetics(fig: Figure, ax: Axes) -> tuple[Figure, Axes]:
    """
    Applies Baseball Savant-style aesthetic formatting to the pitch movement plot.

    Draws a 48x48 inch bounding box, concentric circles at 6-inch intervals,
    and configures the axes limits, tick marks, and gridlines to mimic the
    standard industry pitch movement charts.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The current Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The current Matplotlib axes object.

    Returns
    -------
    tuple[Figure, Axes]
        The updated Matplotlib figure and axes objects with aesthetics applied.
    """

    # draw the 48x48 square box (spanning -24 to 24 in each dimension)
    square = patches.Rectangle((-24, -24), 48, 48, linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(square)

    # center of the square at origin
    center_x, center_y = 0, 0

    # draw concentric circles at 6, 12, 18, and 24 inches; matching savant
    radii = [6, 12, 18, 24]
    for radius in radii:
        circle = patches.Circle(
            (center_x, center_y), radius, linewidth=1.5, edgecolor="black", alpha=0.5, facecolor="none"
        )
        ax.add_patch(circle)

    # set equal aspect ratio and limits
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal")

    # sdd grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Horizontal Break", fontsize=12)
    ax.set_ylabel("Induced Vertical Break", fontsize=12)

    # add tick marks every 6 inches
    ax.set_xticks(range(-24, 25, 6))
    ax.set_yticks(range(-24, 25, 6))

    # add axes through the origin
    ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)

    return fig, ax


def _plot_pitch_movement_legend(fig: Figure, ax: Axes, ncol: int) -> tuple[Figure, Axes]:
    """
    Configures and positions the legend for the pitch movement plot.

    Extracts the existing handles and labels from the axis and mathematically
    reorders them to fill the legend row-wise (left-to-right) rather than the
    Matplotlib default of column-wise (top-to-bottom). Places the legend at the
    lower center of the plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The current Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The current Matplotlib axes object.
    ncol : int
        The number of columns to use in the legend layout.

    Returns
    -------
    tuple[Figure, Axes]
        The updated Matplotlib figure and axes objects with the custom legend.
    """

    # extract the legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # this flips the order from column-wise to row-wise
    handles_flipped = list(itertools.chain(*[handles[i::ncol] for i in range(ncol)]))
    labels_flipped = list(itertools.chain(*[labels[i::ncol] for i in range(ncol)]))

    # draw the legend with the new lists
    ax.legend(handles_flipped, labels_flipped, ncol=ncol, loc="lower center", framealpha=1.0)

    return fig, ax


def _plot_pitch_movement_title(
    fig: Figure, ax: Axes, pitcher: str, date_min: str, date_max: str
) -> tuple[Figure, Axes]:
    """
    Adds formatted titles and subtitles to the pitch movement plot.

    Uses `fig.suptitle` to place the pitcher's name prominently at the top,
    slightly offset to align with the visual center of the plot box, and
    uses `ax.set_title` to display the date range of the queried data just below it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The current Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The current Matplotlib axes object.
    pitcher : str
        The full name of the pitcher.
    date_min : str
        The start date of the data.
    date_max : str
        The end date of the data.

    Returns
    -------
    tuple[Figure, Axes]
        The updated Matplotlib figure and axes objects with titles applied.
    """
    fig.suptitle(pitcher, x=0.532, fontsize=25, fontweight="bold", ha="center")
    ax.set_title(f"{date_min} to {date_max}", fontsize=10)

    return fig, ax


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the pitch movement plot script.

    Sets up defaults for the date range (YTD if mid-season, trailing 365 days otherwise)
    and requires the user to input a pitcher's name.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the attributes `pitcher`, `date_min`,
        and `date_max`.
    """
    today = datetime.now().date()

    parser = argparse.ArgumentParser(description="Make a pitch movement plot for your favorite pitcher")

    parser.add_argument(
        "--pitcher",
        type=str,
        required=True,
        help="Name of the pitcher for whom you want a plot, e.g. 'Caleb Cotham' or 'Cesar Ramos'",
    )

    parser.add_argument(
        "--date_min",
        type=str,
        default=f"{today.year}-01-01" if today.month >= 6 else str(today - timedelta(days=365)),
        help="First date to include in pitch window, 'YYYY-MM-DD' format",
    )

    parser.add_argument(
        "--date_max", type=str, default=str(today), help="Last date to include in pitch window, 'YYYY-MM-DD' format"
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution pipeline for generating a pitch movement plot.

    Parses command line arguments, fetches and processes the data, dynamically builds
    the Matplotlib figure layer by layer (data, aesthetics, titles, legends), and
    displays the final plot.

    Returns
    -------
    None
    """

    args = parse_args()
    pitcher, date_min, date_max = args.pitcher, args.date_min, args.date_max

    # pull the data
    pitch_df_pitcher, pitch_df_pitcher_means = load_and_process_pitch_mvmt_data(
        pitcher=pitcher, date_min=date_min, date_max=date_max
    )

    # make the plots
    fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = _plot_pitch_movement_data(
        fig=fig, ax=ax, pitch_df_pitcher=pitch_df_pitcher, pitch_df_pitcher_means=pitch_df_pitcher_means
    )
    fig, ax = _plot_pitch_movement_aesthetics(fig=fig, ax=ax)
    fig, ax = _plot_pitch_movement_title(fig=fig, ax=ax, pitcher=pitcher, date_min=date_min, date_max=date_max)
    fig, ax = _plot_pitch_movement_legend(fig=fig, ax=ax, ncol=min(pitch_df_pitcher_means.shape[0], 4))

    # display it
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
