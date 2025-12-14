""" """

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_pairwise_relationships(df, hue=None, kind_lower="scatter", corner=True, title=None):
    """
    Generates a mixed-type PairGrid: KDEs on the upper triangle, scatter/reg plots
    on the lower triangle, and histograms on the diagonal.

    This viz allows you to see the correlations and joint 2D densities in one shot. Be
    warned, can take a minute to run on large dataframes

    Parameters
    ----------
    df : pandas.DataFrame
            The input DataFrame. Columns should be numeric.
    hue : str, optional
            Variable in ``df`` to map plot aspects to different colors.
    kind_lower : {'scatter', 'reg'}, optional
            The type of plot to draw on the lower triangle.
            - 'scatter': Simple scatterplot (default).
            - 'reg': Scatterplot with a linear regression line.
    title : str, optional
            Title for the entire figure. Default is None.

    Returns
    -------
    g : seaborn.axisgrid.PairGrid
            The PairGrid object containing the plot.

    Examples
    --------
    >>> import seaborn as sns
    >>> df = sns.load_dataset("iris")
    >>> g = plot_mixed_pairgrid(df, hue='species', kind_lower='reg')
    >>> plt.show()
    """
    # Initialize the PairGrid
    # This creates the empty grid of axes based on the dataframe columns
    grd = sns.PairGrid(df, hue=hue, corner=False)

    # --- Upper Triangle: KDE (Density) ---
    # We use contours (fill=False) to prevent the upper plots from becoming
    # too cluttered if there are many overlapping groups.
    grd.map_upper(sns.kdeplot, fill=False, levels=5, alpha=0.8)

    # --- Lower Triangle: Scatter or Regression ---
    if kind_lower == "reg":
        # Scatter_kws reduces point opacity so you can see the regression line better
        grd.map_lower(
            sns.regplot,
            scatter_kws={"alpha": 0.4, "s": 20},
            line_kws={"color": "black"},
        )
    else:
        grd.map_lower(sns.scatterplot, alpha=0.6, s=20)

    # --- Diagonal: Histograms ---
    grd.map_diag(sns.histplot, kde=True, alpha=0.5)

    # Add a legend if 'hue' is used
    if hue:
        grd.add_legend(title=hue, adjust_subtitles=True)

    # Set title
    if title:
        grd.fig.suptitle(title, y=1.02, fontsize=16)

    return grd


def plot_correlation_matrix(df, title: str = "Correlation Matrix", cmap: str = "coolwarm", annotate: bool = False):
    """
    Generates a lower-triangle correlation heatmap for a pandas DataFrame.

    This function calculates the Pearson correlation coefficient between all pairs
    of numeric columns in the input DataFrame. It then visualizes this matrix
    using a heatmap, masking the upper triangle to reduce visual clutter.

    Parameters
    ----------
    df : pandas.DataFrame
            The input DataFrame containing the data to be analyzed. Columns should
            be numeric.
    title : str, optional
            The title of the plot. Default is 'Correlation Matrix'.
    cmap : str, optional
            The name of the colormap to be used for the heatmap. Default is 'coolwarm',
            which is a diverging colormap suitable for correlation data (-1 to 1).
    annotate : bool, optional
            If True, write the data value in each cell. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
            The Figure object containing the plot.
    ax : matplotlib.axes.Axes
            The Axes object containing the heatmap.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.rand(100, 5), columns=list('ABCDE'))
    >>> fig, ax = plot_correlation_matrix(data, title='My Data Correlations')
    >>> plt.show()
    """
    # Calculate the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    # We use np.triu (triangle upper) to select the upper half and mask it
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    # Adjust figsize based on the number of variables to ensure readability
    num_vars = len(df.columns)
    fig_size = max(8, num_vars * 0.8)  # Dynamic scaling
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=annotate,
        fmt=".2f",
        ax=ax,
    )

    # Aesthetics
    ax.set_title(title, fontsize=16, pad=20)

    # Rotate x-axis labels for better readability if they are long
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Layout adjustment to prevent clipping of tick-labels
    plt.tight_layout()

    return fig, ax
