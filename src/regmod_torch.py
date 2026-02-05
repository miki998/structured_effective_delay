from typing import Callable, Union
from collections import Counter
import numpy as np
import torch
import networkx as nx
from tqdm.notebook import tqdm

def apply_alpha_to_design_torch(
    design_matrix: torch.tensor, alpha: Union[torch.tensor, torch.float]
):
    """Create a design matrix for the path model with already provided optimal path
    matrices and alpha parameter

    Parameters
    ----------
    design_matrix : torch.tensor
        path design matrices.
    alpha : Union[torch.tensor, torch.float] (+ autograd)
        parameter for the sub-optimal paths, by default 0.

    Returns
    -------
    torch.tensor
        design matrix of the path model.
    """

    normalize_vect = torch.zeros(design_matrix.shape[1])
    for i, mat in enumerate(design_matrix):
        if i == 0:
            normalize_vect += torch.sign(mat.sum(axis=1))
        else:
            normalize_vect += torch.sign(mat.sum(axis=1)) * alpha

    normalize_vect[normalize_vect != 0] = 1 / normalize_vect[normalize_vect != 0]

    design_out = torch.zeros_like(design_matrix[0], dtype=torch.float)
    for i in range(len(design_matrix)):
        if i == 0:
            design_out += design_matrix[i]
        else:
            design_out += design_matrix[i] * alpha

    return torch.diag(normalize_vect) @ design_out
