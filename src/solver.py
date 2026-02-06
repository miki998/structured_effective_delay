import sys

from src.regmod_torch import *
import time

def combine_paths_matrices_torch(
    matrices: torch.tensor, alpha: Union[float, torch.tensor] = 0
) -> torch.tensor:
    # NOTE: this function is not optimizable for alpha
    # however this would be a starting point for that
    # so we include it in the solver loop for now
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

    design = torch.zeros_like(matrices[0])
    alpha_id_vector = torch.zeros(design.shape[-1], dtype=int)
    alpha_norm = torch.zeros_like(alpha_id_vector)

    # Compatiblity for the type of alpha
    if isinstance(alpha, (float, int)):
        alpha = torch.tensor([alpha] * len(matrices))
    if isinstance(alpha, (list, tuple)):
        alpha = torch.tensor(alpha)

    if len(alpha) != len(matrices):
        raise ValueError(
            "The alpha parameter must be a scalar or have the same length as the number"
            f" of matrices ({len(alpha)} alphas for {len(matrices)} matrices)."
        )

    for m in matrices:
        # Find rows that have already been filled
        has_shortest_paths = torch.any(design, axis=1)

        # Update the alpha vector for paths that have already been filled
        alpha_vector = has_shortest_paths * alpha[alpha_id_vector] + ~has_shortest_paths
        alpha_id_vector += has_shortest_paths * torch.any(m, axis=1)

        # Update the design matrix
        design += torch.diag(alpha_vector).type(torch.float64) @ m

    # Normalize the design matrix by 1 plus the sum of existing alpha weights
    alpha_norm = torch.tensor([1 + alpha[:i].sum() for i in alpha_id_vector])
    design = torch.diag(1 / alpha_norm).type(torch.float64) @ design
    return design

def forward(a_design: torch.tensor, effective_delay: torch.tensor) -> torch.tensor:
    """
    Computes the estimated delay based on the design matrix and the effective delay.

    Parameters
    ----------
    a_design : torch.tensor
        The design matrix.
    effective_delay : torch.tensor
        The effective delay.

    Returns
    -------
    torch.tensor
        The estimated delay.
    """
    estimated_delay = a_design @ effective_delay
    return estimated_delay

def mse(y1: torch.tensor, y2: torch.tensor) -> torch.tensor:
    """
    Computes the mean squared error (MSE) between two tensors.

    Parameters
    ----------
        y1 (torch.tensor): The first tensor.
        y2 (torch.tensor): The second tensor.
    Returns
    -------
        torch.tensor: The MSE between the two tensors.
    """
    return torch.linalg.norm(y1 - y2)

def pseudo_inverse(y: np.ndarray, a_design: np.ndarray, rcond: float = 1e-15):
    """
    Computes the pseudo-inverse of the input matrix `a_design` and applies it to the
    input vector `y` to obtain the optimal solution `x_opt`.

    Parameters
    ----------
        y (numpy.ndarray): The input vector.
        a_design (numpy.ndarray): The design matrix.
        rcond (float, optional): The relative condition number threshold. Defaults to
        1e-15.

    Returns
    -------
        numpy.ndarray: The optimal solution `x_opt`.
    """

    Ainv = np.linalg.pinv(a_design, rcond=rcond)
    x_opt = Ainv @ y
    return x_opt

def gradient_descent_solver(
    x: torch.tensor,
    y_ground: torch.tensor,
    a_design: torch.tensor,
    delta: float = 0,
    early_stop: float = 1e-5,
    step_size: float = 1e-3,
    l2_penalty: float = 0.1,
    n_iter: int = 1000,
    verbose: bool = False,
    return_logs: bool = False,
) -> tuple[np.ndarray, float]:
    """
    #TODO: note that not a GD anymore
    Performs gradient descent optimization to minimize the mean squared error (MSE)
    between the predicted output `y_pred` and the ground truth `y_ground`.

    Parameters
    ----------
        x (torch.tensor): The input tensor to optimize.
        y_ground (torch.tensor): The ground truth output tensor.
        a_design (torch.tensor): The design matrix.
        delta (float, optional): Value assigned for the synaptic delay parameter.
        early_stop (float, optional): The early stopping threshold. Defaults to 1e-5.
        step_size (float, optional): The step size for gradient descent. Defaults to 1e-3.
        n_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print progress during optimization. Defaults to False.

    Returns
    -------
        tuple[np.ndarray, float]: The optimized input tensor `x_opt` and the final MSE loss.
    """
    nzmask = y_ground > 0 # only optimize for non-zero entries
    y_ground_masked = y_ground[nzmask]
    x.requires_grad = True

    optimizer = torch.optim.Adam([x], lr=step_size)

    df_loss = [-1]
    loss_logs = [-1]
    for _ in tqdm(range(n_iter), total=n_iter, desc="Descent Optimizing...",    file=sys.stdout):
        y_pred = forward(a_design[nzmask], x + delta * (x > 0))

        data_fit = mse(y_pred, y_ground_masked)
        pseudo_fit = torch.linalg.norm(x, ord=2)
        positivity = torch.abs(torch.sum(x * (x < 0).type(torch.float)))
        loss = data_fit + pseudo_fit * l2_penalty + positivity
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_logs.append(loss.item())
        df_loss.append(data_fit.item())

        if torch.diff(torch.tensor(df_loss[-5:])).abs().mean() < early_stop:
            break

    x_opt = x.detach().numpy()
    if return_logs:
        return x_opt, data_fit.item(), loss_logs, df_loss
    return x_opt, data_fit.item()

def gradient_descent_solver_alpha(
    x: torch.tensor,
    y_ground: torch.tensor,
    design: torch.tensor,
    alpha: torch.float,
    delta: float = 0,
    early_stop: float = 1e-5,
    step_size: float = 1e-3,
    l2_penalty: float = 0.1,
    n_iter: int = 1000,
    verbose: bool = False,
    return_logs: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Performs gradient descent optimization to minimize the mean squared error (MSE)
    between the predicted output `y_pred` and the ground truth `y_ground`.

    Parameters
    ----------
        x (torch.tensor): The input tensor to optimize.
        y_ground (torch.tensor): The ground truth output tensor.
        design (torch.tensor): The design matrix.
        delta (float, optional): Value assigned for the synaptic delay parameter.
        early_stop (float, optional): The early stopping threshold. Defaults to 1e-5.
        step_size (float, optional): The step size for gradient descent. Defaults to 1e-3.
        n_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print progress during optimization. Defaults to False.

    Returns
    -------
        tuple[np.ndarray, float]: The optimized input tensor `x_opt` and the final MSE loss.
    """
    nzmask = y_ground > 0 # only optimize for non-zero entries
    y_ground_masked = y_ground[nzmask]
    x.requires_grad = True
    alpha.requires_grad = True

    optimizer = torch.optim.Adam([x, alpha], lr=step_size)

    df_loss = [-1]
    loss_logs = [-1]
    for i in tqdm(range(n_iter), disable=not verbose, desc="Descent(Alpha) Optimizing..."):
        # start = time.time()
        a_design = apply_alpha_to_design_torch_accelerate(design, alpha)
        # print(f"Time for design matrix computation accelerated: {time.time() - start:.4f} seconds")
        # start = time.time()
        # a_design2 = apply_alpha_to_design_torch(design, alpha)
        # print(f"Time for design matrix computation original: {time.time() - start:.4f} seconds")

        # print((a_design.type(torch.float) - a_design2.type(torch.float)).min())
        # assert torch.allclose(a_design.type(torch.float), a_design2.type(torch.float), atol=1e-6), "Mismatch between accelerated and original implementation"

        a_design = a_design[nzmask].type(torch.float)
        

        y_pred = forward(a_design, x + delta * (x > 0))

        data_fit = mse(y_pred, y_ground_masked)
        pseudo_fit = torch.linalg.norm(x, ord=2)
        positivity = torch.abs(torch.sum(x * (x < 0).type(torch.float)))
        loss = data_fit + pseudo_fit * l2_penalty + positivity
        loss.backward()

        # print(np.isnan(a_design.detach().numpy()).any())
        # print(np.isnan((x + delta * (x > 0)).detach().numpy()).any())

        # print("\n######")
        # print("pred", y_pred.detach().numpy())
        # print("loss", loss.detach().item(), "alpha", alpha.detach().item())
        # print("alpha.grad", alpha.grad.detach())
        # print("x.grad", np.isnan(x.grad.detach().numpy()).any())
        # print("alpha", alpha.detach())

        if alpha.grad is not None:
            alpha.grad.nan_to_num_(nan=0.0)
        if x.grad is not None:
            x.grad.nan_to_num_(nan=0.0)

        optimizer.step()
        optimizer.zero_grad()

        loss_logs.append(loss.item())
        df_loss.append(data_fit.item())

        if torch.diff(torch.tensor(df_loss[-5:])).abs().mean() < early_stop:
            print(f"Stopped at iteration #{i}")
            break

    x_opt = x.detach().numpy()
    a_opt = alpha.detach().item()

    if return_logs:
        return x_opt, a_opt, data_fit.item(), loss_logs, df_loss

    return x_opt, a_opt, data_fit.item()

def gradient_descent_solver_delta(
    x: torch.tensor,
    y_ground: torch.tensor,
    a_design: torch.tensor,
    delta: float = 0,
    early_stop: float = 1e-5,
    step_size: float = 1e-3,
    l2_penalty: float = 0.1,
    n_iter: int = 1000,
    verbose: bool = False,
    return_logs: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Performs gradient descent optimization to minimize the mean squared error (MSE)
    between the predicted output `y_pred` and the ground truth `y_ground`.

    Parameters
    ----------
        x (torch.tensor): The input tensor to optimize.
        y_ground (torch.tensor): The ground truth output tensor.
        a_design (torch.tensor): The design matrix (+ with autograd)
        delta (float, optional): Value assigned for the synaptic delay parameter.
        early_stop (float, optional): The early stopping threshold. Defaults to 1e-5.
        step_size (float, optional): The step size for gradient descent. Defaults to 1e-3.
        n_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print progress during optimization. Defaults to False.

    Returns
    -------
        tuple[np.ndarray, float]: The optimized input tensor `x_opt` and the final MSE loss.
    """
    nzmask = y_ground > 0 # only optimize for non-zero entries
    y_ground_masked = y_ground[nzmask]
    x.requires_grad = True
    delta.requires_grad = True
    
    optimizer = torch.optim.Adam([x, delta], lr=step_size)

    df_loss = [-1]
    loss_logs = [-1]
    for i in tqdm(range(n_iter), disable=not verbose, desc="Descent(Delta) Optimizing..."):
        y_pred = forward(a_design[nzmask], x + delta * (x > 0))

        data_fit = mse(y_pred, y_ground_masked)
        pseudo_fit = torch.linalg.norm(x, ord=2)
        positivity = torch.abs(torch.sum(x * (x < 0).type(torch.float)))
        delta_positivity = 10 * torch.abs(delta * (delta < 0).type(torch.float))

        loss = data_fit + pseudo_fit * l2_penalty + positivity + delta_positivity
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_logs.append(loss.item())
        df_loss.append(data_fit.item())

        if torch.diff(torch.tensor(df_loss[-5:])).abs().mean() < early_stop:
            print(f"Stopped at iteration #{i}")
            break

    x_opt = x.detach().numpy()
    delta_opt = delta.detach().item()
    if return_logs:
        return x_opt, delta_opt, data_fit.item(), loss_logs, df_loss

    return x_opt, delta_opt, data_fit.item()

def effective_delay_solver(
    x: torch.tensor,
    y_ground: torch.tensor,
    design: torch.tensor,
    alpha: torch.float,
    delta: float = 0,
    early_stop: float = 1e-5,
    step_size: float = 1e-3,
    l2_penalty: float = 0.1,
    n_iter: int = 1000,
    verbose: bool = False,
    return_logs: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Performs gradient descent optimization to minimize the mean squared error (MSE)
    between the predicted output `y_pred` and the ground truth `y_ground`.
    Jointly optimizes for alpha and delta.

    Parameters
    ----------
        x (torch.tensor): The input tensor to optimize.
        y_ground (torch.tensor): The ground truth output tensor.
        a_design (torch.tensor): The design matrix (+ with autograd)
        delta (float, optional): Value assigned for the synaptic delay parameter.
        early_stop (float, optional): The early stopping threshold. Defaults to 1e-5.
        step_size (float, optional): The step size for gradient descent. Defaults to 1e-3.
        n_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print progress during optimization. Defaults to False.

    Returns
    -------
        tuple[np.ndarray, float]: The optimized input tensor `x_opt` and the final MSE loss.
    """
    nzmask = y_ground > 0 # only optimize for non-zero entries
    y_ground_masked = y_ground[nzmask]
    x.requires_grad = True
    alpha.requires_grad = True
    delta.requires_grad = True

    optimizer = torch.optim.Adam([x, alpha, delta], lr=step_size)

    df_loss = [-1]
    loss_logs = [-1]
    for i in tqdm(range(n_iter), disable=not verbose, desc="Descent(Joint) Optimizing..."):
        a_design = apply_alpha_to_design_torch_accelerate(design, alpha)
        a_design = a_design[nzmask].type(torch.float)

        y_pred = forward(a_design, x + delta * (x > 0))
        data_fit = mse(y_pred, y_ground_masked)
        pseudo_fit = torch.linalg.norm(x, ord=2)
        positivity = torch.abs(torch.sum(x * (x < 0).type(torch.float)))

        loss = data_fit + pseudo_fit * l2_penalty + positivity
        loss.backward()

        if alpha.grad is not None:
            alpha.grad.nan_to_num_(nan=0.0)
        if delta.grad is not None:
            delta.grad.nan_to_num_(nan=0.0)
        if x.grad is not None:
            x.grad.nan_to_num_(nan=0.0)

        optimizer.step()
        optimizer.zero_grad()

        loss_logs.append(loss.item())
        df_loss.append(data_fit.item())

        if torch.diff(torch.tensor(df_loss[-5:])).abs().mean() < early_stop:
            print(f"Stopped at iteration #{i}")
            break

    x_opt = x.detach().numpy()
    alpha_opt = alpha.detach().item()
    delta_opt = delta.detach().item()
    if return_logs:
        return x_opt, (alpha_opt, delta_opt), data_fit.item(), loss_logs, df_loss
    
    return x_opt, (alpha_opt, delta_opt), data_fit.item()
