from typing import Union
import numpy as np
import scipy.io as sio
import scipy.sparse.csgraph as csgraph


def evaluate(input_file: str, solution_file: str) -> Union[int, float]:
    """
    Evaluator for the Graph Drawing problem.
    Computes the total stress value based on node coordinates output by the solver.

    Args:
        input_file: Path to the input .mat file (containing Problem.A adjacency matrix)
        solution_file: Path to the solver output file (node coordinates)

    Returns:
        float: Total stress value (lower is better)
    """

    try:
        data = sio.loadmat(input_file)
        A = data["Problem"]["A"][0][0]
    except Exception as e:
        raise ValueError(f"Error loading adjacency matrix: {e}")

    n = A.shape[0]

    D = csgraph.shortest_path(A, directed=False, unweighted=True)
    # Some pairs might be disconnected -> set them as NaN (ignored in stress)
    D[D == np.inf] = np.nan

    # Compute weights w_ij = 1 / d_ij^2
    W = np.zeros_like(D)
    mask = ~np.isnan(D) & (D > 0)
    W[mask] = 1.0 / (D[mask] ** 2)

    try:
        coords = np.loadtxt(solution_file, usecols=(1, 2))
    except Exception as e:
        raise ValueError(f"Error loading coordinates: {e}")

    if coords.shape[0] != n:
        raise ValueError(
            f"Number of nodes in output ({coords.shape[0]}) "
            f"does not match adjacency matrix ({n})."
        )

    # --- Compute stress ---
    stress = 0.0
    for i in range(n):
        for j in range(i):
            if np.isnan(D[i, j]) or D[i, j] == 0:
                continue
            dist_ij = np.linalg.norm(coords[i] - coords[j])
            stress += W[i, j] * (dist_ij - D[i, j]) ** 2

    return float(stress)
