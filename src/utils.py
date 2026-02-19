import pickle
import numpy as np
import matplotlib.pyplot as plt
import json


def load_json(json_filename: str):
    """
    Load an object saved by save_json. Reconstructs numpy arrays, pandas DataFrames/Series,
    and torch.Tensors (if torch is available).

    Parameters
    ----------
    json_filename : str
        Path to the JSON file to load.

    Returns
    -------
    Any
        The reconstructed Python object.
    """

    with open(json_filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save(pickle_filename: str, iterable: object) -> None:
    """
    Pickle an object to a file.

    Parameters
    ----------
    pickle_filename : str
        Path to the file where the object will be pickled.
    iterable : object
        The object to be pickled.

    Returns
    -------
    None
    """
    with open(pickle_filename, "wb") as handle:
        pickle.dump(iterable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(pickle_filename: str) -> object:
    """
    Load a pickled object from the specified file.

    Parameters
    ----------
    pickle_filename : str
        The filename of the pickled object to load.

    Returns
    -------
    object
        The loaded object.
    """
    with open(pickle_filename, "rb") as handle:
        b = pickle.load(handle)
    return b

def add_cbar(fig, ax, ticksize=10, **kwargs):
    """Add a colorbar to an existing figure/axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to get the colorbar from
    ax : matplotlib.axes.Axes
        axes to add the colorbar to

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        tuple of figure and axes with the colorbar added
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical", **kwargs)
    cbar.ax.tick_params(labelsize=ticksize)
    return fig, ax


def annotate_heatmap(
    fig: plt.Figure, ax: plt.Axes, matrix: np.ndarray, adapt_color: float = 0
) -> tuple[plt.Figure, plt.Axes]:
    """Annotate heatmap with values from a matrix.

    Parameters
    ----------
    fig : plt.Figure
        figure to annotate
    ax : plt.Axes
        axis to annotate
    matrix : np.ndarray
        data matrix
    adapt_color : float, optional
        threshold value to alternate from dark (above) to light (below) text, by
        default 0

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        figure and axes with the heatmap annotated
    """

    for row_i, row in enumerate(matrix):
        for col_j, val in enumerate(row):

            color = "w" if val < adapt_color else "k"
            if val > 0:
                ax.text(
                    row_i, col_j, f"{val:1.2f}", ha="center", va="center", color=color
                )
    return fig, ax

def remove_diagonal_entries(A: np.ndarray) -> np.ndarray:
    """
    Remove the diagonal entries from the input matrix `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The input matrix.

    Returns
    -------
    numpy.ndarray
        The input matrix `A` with the diagonal entries removed.
    """
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)

def add_diagonal_entries(A: np.ndarray) -> np.ndarray:
    """
    Add diagonal entries with zeros to a 2D numpy array.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D numpy array with shape (d-1, d).

    Returns
    -------
    numpy.ndarray
        A 2D numpy array with shape (d, d) where the diagonal entries are set to 0.
    """
    d = A.shape[0]
    assert A.shape[1] == d - 1
    ret = np.ndarray((d, d+1), dtype=A.dtype)
    ret[:,0] = 0
    ret[:-1,1:] = A.reshape((d-1, d))
    ret = ret.reshape(-1)[:-d].reshape(d,d)
    return ret