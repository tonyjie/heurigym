"""
the baseline function
"""
from typing import Union
import scipy.io as sio
import scipy.sparse.csgraph as csgraph
import numpy as np
from collections import deque
import math
import random
import time
import s_gd2 as sgd2
import os

############################################################
# 1. Stress evaluator
############################################################

def compute_sgd_stress(A, X):
    D = csgraph.shortest_path(A, directed=False, unweighted=True)
    n = A.shape[0]
    stress = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dij = D[i, j]
            if np.isinf(dij) or dij == 0:
                continue
            w = 1.0 / (dij ** 2)
            dist = np.linalg.norm(X[i] - X[j])
            stress += w * (dist - dij) ** 2
    return stress

############################################################
# 2. Graph utilities (unweighted)
############################################################

class Term:
    def __init__(self, i, j, d, w):
        self.i = i
        self.j = j
        self.d = d
        self.w = w


def build_graph_unweighted(n, I, J):
    graph = [[] for _ in range(n)]
    undirected = [set() for _ in range(n)]
    for i, j in zip(I, J):
        if i == j:
            continue
        if j not in undirected[i] and i not in undirected[j]:
            undirected[i].add(j)
            undirected[j].add(i)
            graph[i].append(j)
            graph[j].append(i)
    return graph


def bfs_terms(n, I, J):
    graph = build_graph_unweighted(n, I, J)
    terms = []
    for source in range(n - 1):
        d = [-1] * n
        d[source] = 0
        q = deque([source])
        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if d[nxt] == -1:
                    d[nxt] = d[cur] + 1
                    q.append(nxt)
                    if source < nxt:
                        dij = d[nxt]
                        wij = 1.0 / (dij * dij)
                        terms.append(Term(source, nxt, dij, wij))
    return terms

############################################################
# 3. Learning‑rate schedule
############################################################

def schedule(terms, t_max, eps):
    w_min = min(t.w for t in terms)
    w_max = max(t.w for t in terms)
    eta_max = 1.0 / w_min
    eta_min = eps / w_max
    lamb = math.log(eta_max / eta_min) / (t_max - 1)
    return [eta_max * math.exp(-lamb * t) for t in range(t_max)]

############################################################
# 4. SGD core
############################################################

def sgd(X, terms, etas, seed, time_limit=None):
    rng = random.Random(seed)
    start_time = time.time()
    for eta in etas:
        rng.shuffle(terms)
        if time_limit is not None and time.time() - start_time > time_limit:
            return
        for t in terms:
            if time_limit is not None and time.time() - start_time > time_limit:
                return
            i, j = t.i, t.j
            w_ij = t.w
            d_ij = t.d
            mu = eta * w_ij
            if mu > 1:
                mu = 1
            dx = X[i, 0] - X[j, 0]
            dy = X[i, 1] - X[j, 1]
            mag = math.sqrt(dx * dx + dy * dy)
            r = (mu * (mag - d_ij)) / (2 * mag)
            rx = r * dx
            ry = r * dy
            X[i, 0] -= rx
            X[i, 1] -= ry
            X[j, 0] += rx
            X[j, 1] += ry

############################################################
# 5. High‑level unweighted layout
############################################################

def layout_unweighted_py(X, I, J, t_max, eps, seed, time_limit=None):
    n = X.shape[0]
    terms = bfs_terms(n, I, J)
    etas = schedule(terms, t_max, eps)
    sgd(X, terms, etas, seed, time_limit=time_limit)
    return X

############################################################
# 6. Initialization utilities
############################################################

def _check_random_seed(random_seed=None):
    if random_seed is None:
        random_seed = np.random.randint(65536)
    return random_seed


def _random_init(n, random_seed, init=None, num_dimensions=2):
    if init is not None:
        X = np.ascontiguousarray(init)
        return X
    np.random.seed(random_seed)
    if num_dimensions == 2:
        return np.random.rand(n, 2)
    raise ValueError("only 2D layouts supported")


def random_init(I, J, random_seed, init=None):
    n = max(max(I), max(J)) + 1
    return _random_init(n, random_seed, init)

############################################################
# 7. Public solver entry (final API)
############################################################

def sgd_solve(input_file: str, time_limit=120):
    # timelimit保持和timeout的一样；但保留接口
    time_limit = int(os.getenv("SOLVER_TIMEOUT", 120))
    try:
        mat_data = sio.loadmat(input_file)
    except Exception:
        raise ValueError(f"Unable to read {input_file}")

    A = mat_data['Problem']['A'][0][0]
    A = A + A.T
    A[A > 0] = 1

    from scipy.sparse import triu
    A_upper = triu(A, k=1)
    I, J = A_upper.nonzero()

    n = A.shape[0]
    X = random_init(I, J, random_seed=12)
    X = layout_unweighted_py(X, I, J, t_max=30, eps=0.01, seed=12, time_limit=time_limit)
    stress = compute_sgd_stress(A, X)
    return I, J, X, stress, n

def sgd_origin_solve(input_file: str, time_limit=120):
    """
    Use the original C++ implementation from the SGDW paper.
    """
    try:
        mat_data = sio.loadmat(input_file)
    except Exception as e:
        raise ValueError(f"Unable to download {input_file}. ")

    #### get A
    A = mat_data['Problem']['A'][0][0]
    A = A + A.T  #### makesure A is symmetric
    A[A > 0] = 1

    #### get connected coordinates (I, J)
    from scipy.sparse import triu
    A_upper = triu(A, k=1)
    I, J = A_upper.nonzero()
    n = A.shape[0]

    #### compute graph layout (x,y)
    # X = sgd_layout(I, J, random_seed=12, time_limit=time_limit)
    X = sgd2.layout(I, J, random_seed=12)
    stress = compute_sgd_stress(A, X)
    # print("SGD Stress:", stress)
    return I, J, X, stress, n

def _draw_png_matplotlib(
    X, I, J, filepath=None,
    noderadius=0.2, linkwidth=0.05,
    nodeopacity=1, linkopacity=1,
    ax=None,
):
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))



    # ==== 固定比例和范围 ====
    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_box_aspect(1)


    # ==== 绘制边 + 节点 ====
    links = zip((X[i] for i in I), (X[j] for j in J))
    lc = mc.LineCollection(links, linewidths=linkwidth, colors=(0, 0, 0, linkopacity))
    ax.add_collection(lc)

    ax.scatter(X[:, 0], X[:, 1], s=1.5, color="black", alpha=nodeopacity)
    ax.margins(0.2)

    ax.axis("off")

    if filepath:
        ax.figure.savefig(filepath, dpi=200, bbox_inches="tight", format="png", transparent=True)
