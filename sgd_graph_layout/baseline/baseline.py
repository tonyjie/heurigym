import scipy.io as sio
import numpy as np
import random
import scipy.sparse.csgraph as csgraph
import s_gd2 as sgd
import os
import json

def compute_sgd_stress(A, X):
    """
    Compute layout stress given adjacency matrix A and coordinates X.

    Args:
        A: (n x n) adjacency matrix (numpy array or sparse)
        X: (n x 2) coordinates of nodes

    Returns:
        float: total stress
    """
    # Compute shortest-path distances
    D = csgraph.shortest_path(A, directed=False, unweighted=True)
    n = A.shape[0]

    stress = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dij = D[i, j]
            if np.isinf(dij) or dij == 0:
                continue  # skip disconnected pairs or self-loops
            w = 1.0 / (dij ** 2)
            dist = np.linalg.norm(X[i] - X[j])
            stress += w * (dist - dij) ** 2
    return stress

def sgd_solve(input_file: str):
    
    try:
        mat_data = sio.loadmat(input_file)
    except Exception as e:
        raise ValueError(f"Unable to download {input_file}. ")

    #### get A
    A = mat_data['Problem']['A'][0][0]  
    A = A + A.T  #### makesure A is symmetric
    # A[A < 0] = 0
    A[A > 0] = 1
    
    #### get connected coordinates (I, J) 
    from scipy.sparse import triu
    A_upper = triu(A, k=1)
    I, J = A_upper.nonzero() 

    #### compute graph layout (x,y)
    X = sgd.layout(I, J, t_max=50, random_seed=12)
    stress = compute_sgd_stress(A, X)
    # print("SGD Stress:", stress)
    return I, J, X, stress


if __name__ == "__main__":
    test_dir = "selfneeded/test_dataset"
    results = {}

    for fname in os.listdir(test_dir):
        if not fname.endswith(".mat"):
            continue

        input_file = os.path.join(test_dir, fname)
        svg_name = os.path.splitext(os.path.basename(input_file))[0]
        print(f"Processing {svg_name}...")

        try:
            I, J, sgd_layout, sgd_stress = sgd_solve(input_file)
            results[svg_name] = float(sgd_stress)
            print(f"✅ {svg_name}: stress = {sgd_stress:.6f}")
        except Exception as e:
            print(f"❌ {svg_name} failed: {e}")
            # results[svg_name] = 1e12

    with open("baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("baseline.json updated.")
