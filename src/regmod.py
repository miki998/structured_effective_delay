from typing import Callable, Union
from collections import Counter
import numpy as np
from scipy import linalg
import networkx as nx
from tqdm.notebook import tqdm


def build_design_binary(adjacency: np.ndarray):
    """Create a design matrix from the structural connectivity matrix. Each row
    represents a connection (e.g. delay) between two nodes and each column is an edge
    of the structural connectome. There is a 1 a structural connectome edge is going
    connected to the "sending" node.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the structural connectivity.

    Returns
    -------
    np.ndarray
        binary design matrix of the degree model.
    """
    n = adjacency.shape[0]

    blocks = [np.vstack([np.delete(adjacency[i], i)] * (n - 1)) for i in range(n)]

    a_binary = linalg.block_diag(*blocks)
    return a_binary


def build_design_degree(adjacency: np.ndarray, target_deg: bool = False):
    """Create a design matrix similar to the binary model but with weights. Weights are
    dependent on the degree of the "sending" node, the "receiving" node and nodes
    connected to the "sending" node (see `target_deg` parameter for details).

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the structural connectivity.
    target_deg : bool, optional
        condition to use the degree of the "receiving" node instead of the degrees of
        nodes connected to the "sending" node as the numerator of the weight (see
        formula below), by default False.

        True: w_ij = deg(i) / (deg(i) + deg(j))
        False: w_ij = deg(j) / (deg(i) + deg(j))

    Returns
    -------
    np.ndarray
        weighted design matrix of the degree model.
    """
    n = adjacency.shape[0]

    a_binary = build_design_binary(adjacency)

    degrees = adjacency.sum(axis=1)

    nominator = np.ones((len(degrees), 1)) * degrees

    if target_deg:
        nominator = nominator.T

    denum = degrees + degrees.reshape((-1, 1))
    weights = np.divide(nominator, denum)

    blocks = [np.delete(np.delete(weights, i, axis=0), i, axis=1) for i in range(n)]
    a_weights = linalg.block_diag(*blocks)
    return a_binary * a_weights


def get_normalize_function(normalization: str = "length") -> Callable:
    """Helper to get the normalization function based on the conditions provided.

    Parameters
    ----------
    normalization : str, optional
        type of normalization, by default "length". The options are:
        - (Default) normalization by the number of paths of each length
        - normalization by total number of paths
        - no normalization


    Returns
    -------
    Callable
        normalization function.
    """

    if normalization is None:

        def norm_func(x):
            return 1

        return norm_func

    if "total" in normalization:
        norm_func = np.sum
    else:

        def norm_func(x):
            return x

    return norm_func


def get_path_matrices(
    adjacency: np.ndarray, max_path_length: int = 2, normalization: str = "length"
):
    """Computes a design matrix for each path length of the path model. It has a value
    of k if the edge on column j is in a path of length k between the two nodes
    connected by the edge on row i. The maximum path length is defined by the
    `max_path_length` parameter.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the graph.
    max_path_length : int, optional
        maximum path length allowed in the search, by default 2
    normalization : str, optional
        type of normalization, by default "length". The options are:
        - (Default) normalization by the number of paths of each length
        - normalization by total number of paths
        - no normalization

    Returns
    -------
    np.ndarray
        design matrices of the path model sorted in ascending path length.
    """

    s_graph = nx.Graph(adjacency)
    n = len(adjacency)
    n_edges = n * (n - 1)

    edges_id = [(i, j) for i in range(n) for j in range(n) if i != j]
    edge_to_edge_id_dict = {ed: i for i, ed in enumerate(edges_id)}
    edges_id = np.array(edges_id)

    a_design = np.zeros((max_path_length, n_edges, n_edges))
    norm_weights = np.array([np.eye(n_edges) for _ in range(max_path_length)])

    norm_func = get_normalize_function(normalization)

    for design_i, (node_in, node_out) in enumerate(tqdm(edges_id)):
        all_paths = nx.all_simple_edge_paths(
            s_graph, node_in, node_out, cutoff=max_path_length
        )
        n_path_per_length = np.zeros(max_path_length)
        for path in all_paths:
            path_len = len(path)
            n_path_per_length[path_len - 1] += 1
            ids = [edge_to_edge_id_dict[coords] for coords in path]

            a_design[path_len - 1, design_i, ids] += 1

        norm_weights[:, design_i, design_i] = norm_func(n_path_per_length)

    norm_weights = np.divide(
        1, norm_weights, out=np.zeros_like(norm_weights), where=norm_weights != 0
    )
    a_design = norm_weights @ a_design

    return a_design


def combine_paths_matrices(
    matrices: np.ndarray, alpha: Union[float, np.ndarray] = 0
) -> np.ndarray:
    """Create a design matrix for the path model by combining the design matrices of
    each path lengths.

    Parameters
    ----------
    matrices : np.ndarray
        individual design matrices for each path length
    alpha : Union[float, np.ndarray], optional
        hyperparameter to include the influence of sub-optimal paths (could be one
        single value or a value for length greater than the shortest path), by default 0

    Returns
    -------
    np.ndarray
        design matrix of the path model.

    Raises
    ------
    ValueError
        the `alpha` parameter should be a scalar or have the same length as the number
        of matrices.
    """

    design = np.zeros_like(matrices[0])
    alpha_id_vector = np.zeros(design.shape[-1], dtype=int)
    alpha_norm = np.zeros_like(alpha_id_vector)

    # Compatiblity for the type of alpha
    if isinstance(alpha, (float, int)):
        alpha = np.array([alpha] * len(matrices))
    if isinstance(alpha, (list, tuple)):
        alpha = np.array(alpha)

    if len(alpha) != len(matrices):
        raise ValueError(
            "The alpha parameter must be a scalar or have the same length as the number"
            f" of matrices ({len(alpha)} alphas for {len(matrices)} matrices)."
        )

    for m in matrices:
        # Find rows that have already been filled
        has_shortest_paths = np.any(design, axis=1)

        # Update the alpha vector for paths that have already been filled
        alpha_vector = has_shortest_paths * alpha[alpha_id_vector] + ~has_shortest_paths
        alpha_id_vector += has_shortest_paths * np.any(m, axis=1)

        # Update the design matrix
        design += np.diag(alpha_vector) @ m

        # Early stopping if all rows have been filled and alpha is zero
        if np.any(design, axis=1).all() and np.isclose(alpha, 0).all():
            print("Early stopping !")
            return design

    # Normalize the design matrix by 1 plus the sum of existing alpha weights
    alpha_norm = np.array([1 + alpha[:i].sum() for i in alpha_id_vector])
    design = np.diag(1 / alpha_norm) @ design
    return design


def get_shortest_matrices(
    adjacency: np.ndarray, n_subopt: int = 0, progress: bool = True
):
    """Create design matrices for level of suboptimal shortest paths.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the graph.
    n_subopt : int, optional
        number of sub-optimal path to consider, by default 0.
    progress : bool, optional
        condition to show a progress bar, by default True.

    Returns
    -------
    np.ndarray
        design matrices for each level of suboptimal paths.
    """

    graph = nx.Graph(adjacency)

    n_nodes = graph.number_of_nodes()
    all_length = dict(nx.shortest_path_length(graph))
    all_nodes_pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    edge_to_edge_id_dict = {ed: i for i, ed in enumerate(all_nodes_pairs)}

    design_matrix = np.zeros((1 + n_subopt, len(all_nodes_pairs), len(all_nodes_pairs)))

    if progress:
        tqdm_fun = tqdm
    else:

        def tqdm_fun(x, *args, **kwargs):
            return x

    for row_i, (i, j) in enumerate(
        tqdm_fun(all_nodes_pairs, total=len(all_nodes_pairs))
    ):
        max_length = all_length[i][j] + n_subopt
        paths = list(nx.all_simple_edge_paths(graph, i, j, cutoff=max_length))

        length_count = Counter([len(p) for p in paths])

        # Looping is faster than using `map` or concatenating to a list of edges
        for p in paths:
            len_id = len(p) - all_length[i][j]
            for e in p:
                # design_matrix[len_id, row_i, edge_to_edge_id_dict[e]] += 1 / len(p)
                design_matrix[len_id, row_i, edge_to_edge_id_dict[e]] += (
                    1 / length_count[len(p)]
                )

    return design_matrix


def apply_alpha_to_design(
    design_matrix: np.ndarray,
    n_subopt: int = 0,
    alpha: Union[list, np.ndarray, int, float] = 0,
):
    """Create a design matrix for the path model with already provided optimal path
    matrices and alpha parameter

    Parameters
    ----------
    design_matrix : np.ndarray
        path design matrices.
    n_subopt : int, optional
        number of sub-optimal path to consider, by default 0.
    alpha : Union[list, np.ndarray, int, float], optional
        parameter for the sub-optimal paths, by default 0. If a scalar (or float), the
        alpha value will be broadcasted to all rows of the design matrix. If an array,
        it is required to have the same size as the number of rows in the design matrix
        (row-wise alpha). If a list, the alpha value will be applied to each
        sub-optimal path.

    Returns
    -------
    np.ndarray
        design matrix of the path model.
    """

    # Compatibility for int or float alpha
    if isinstance(alpha, (float, int)):
        a_rowise = np.array([alpha] * design_matrix.shape[1])

    # Compatibility for row-wise alphas (np.ndarray)
    elif len(alpha) == design_matrix.shape[1]:
        a_rowise = alpha.copy()
    elif len(alpha) != n_subopt:
        raise ValueError(
            "The alpha parameter must be a scalar, a list of size equal to the number"
            f" of sub-optimal paths ({n_subopt}) or a np.ndarray of size equal to the"
            f" number of edges ({design_matrix.shape[1]}). Size is {len(alpha)}."
        )

    # Compatibility for list of row-wise alphas (one per sub-optimal path)
    if isinstance(alpha, list):
        a_vec = alpha.copy()
    else:
        a_vec = [np.ones_like(a_rowise)] + [a_rowise] * n_subopt

    # Vectors of alphas if (sub-)optimal paths exist
    normalize_vect = np.array(
        [np.sign(mat.sum(axis=1)) * a_vec[i] for i, mat in enumerate(design_matrix)]
    )
    normalize_vect = sum(normalize_vect)
    normalize_vect = np.divide(1, normalize_vect, where=normalize_vect != 0)
    
    # Compute design matrix (without normalization)
    design_out = np.sum(
        [np.diag(a) @ design_matrix[i] for i, a in enumerate(a_vec)], axis=0
    )

    return np.diag(normalize_vect) @ design_out


def build_design_shortest(
    adjacency: np.ndarray, n_subopt: int = 0, alpha: Union[float, list] = 0
) -> np.ndarray:
    """Create a design matrix for the path model using only the shortest paths. This
    implementation is faster than parsing through all paths (relevant for alpha = 0).

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the graph.
    n_subopt : int, optional
        number of sub-optimal path to consider, by default 0.
    alpha : Union[float, list], optional
        parameter for the sub-optimal paths, by default 0.

    Returns
    -------
    np.ndarray
        design matrix of the path model.
    """

    design_matrix = get_shortest_matrices(adjacency=adjacency, n_subopt=n_subopt)
    out_design = apply_alpha_to_design(design_matrix, n_subopt=n_subopt, alpha=alpha)

    return out_design


def build_design_paths_old(adjacency: np.ndarray, alpha: float, **kwargs) -> np.ndarray:
    """Build the design matrix of the path model by summing the design matrices of each
    path length, weighted by powers the `alpha` parameter.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the graph.
    alpha : float
        parameter to weight the design matrices of paths with length greater than one.

    Returns
    -------
    np.ndarray
        design matrix of the path model.
    """

    matrices = get_path_matrices(adjacency, **kwargs)

    a_design_path = np.sum(
        [np.power(alpha, i) * a_design_i for i, a_design_i in enumerate(matrices)],
        axis=0,
    )

    return a_design_path


def predict_conduction_delays(
    a_design: np.ndarray,
    x: np.ndarray,
    is_matrix: bool = True,
    invert_weights: bool = False,
) -> np.ndarray:
    """Predict the conduction delays from provided effective delay `x` by multiplying
    with the design matrix `a_design`. The effective delay `x` can be provided as a
    vector or as a matrix given the condition in `is_matrix`.

    Parameters
    ----------
    a_design : np.ndarray
        design matrix of the regression model.
    x : np.ndarray
        effective delay to be predicted (either a vector of length n_edges or a matrix
        of shape n_nodes x n_nodes).
    is_matrix : bool, optional
        condition to provide `x` and to return `y_hat` as matrices, by default True
    invert_weights : bool, optional
        use the non-zero weights of the design matrix as one over the original weight,
        by default False

    Returns
    -------
    np.ndarray
        predicted conduction delays.
    """

    x_pred = x.copy()
    if is_matrix:
        off_diag_ids = np.array(
            [(i, j) for i in range(len(x)) for j in range(len(x)) if i != j]
        ).T

        # x_pred = x[*off_diag_ids]
        x_pred = x[off_diag_ids[0], off_diag_ids[1]]

    a_predict = a_design.copy()
    if invert_weights:
        a_predict = np.divide(
            1,
            a_design,
            out=np.zeros_like(a_design, dtype=float),
            where=a_design != 0,
        )

    y_pred = a_predict @ x_pred

    if is_matrix:
        y_pred_mat = np.zeros_like(x)

        # y_pred_mat[*off_diag_ids] = y_pred
        y_pred_mat[off_diag_ids[0], off_diag_ids[1]] = y_pred

        return y_pred_mat
    return y_pred
