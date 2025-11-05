# Graph Drawing by Stochastic Gradient Descent (SGD)

## Background

Graphs are a common data structure used to describe everything. Any set of pairwise relationships between entities can be represented as a graph, and the growing amount of data collected makes visualizing graphs for exploratory analysis an important task. Node-link diagrams provide an intuitive representation, where vertices are shown as dots and edges as lines connecting them. The main task is to find suitable coordinates for these dots that represent the data faithfully. 

Multidimensional scaling (MDS) addresses this problem by minimizing the disparity between ideal and low-dimensional distances through an error function called stress. Kamada and Kawai minimized this function with a localized Newton-Raphson method, while Kruskal and later De Leeuw used gradient descent and majorization respectively. The latter, applied to graph layout by Gansner et al., has remained a standard approach.

A new method that minimizes stress uses stochastic gradient descent (SGD). The algorithm moves a single pair of vertices at a time, approximating the global gradient through local updates. Its stochastic nature helps avoid local minima, achieving lower stress faster and more consistently than majorization without requiring careful initialization. Moreover, the simplicity of SGD makes it well suited for constrained layouts and scalable to large graphs through sparse stress approximations.

## Formalization

**Definitions:**

The graph layout problem can be formulated as minimizing the stress function used in multidimensional scaling (MDS).

The problem is to compute 2D coordinates for all nodes in a graph so that the Euclidean distances between nodes match their graph-theoretic shortest path distances as closely as possible. It produces a visually meaningful layout of the graph.

**Objective:**

Let $X = [X_1, X_2, \ldots, X_n]$ denote the coordinates of all vertices in low-dimensional space, where $X_i \in \mathbb{R}^d$ is the position of vertex $i$. Let $d_{ij}$ be the ideal graph-theoretic (shortest path) distance between vertices $i$ and $j$, and $w_{ij}$ be a weighting factor, typically $w_{ij} = d_{ij}^{-2}$.

The objective is to minimize the stress function:

$$
\text{stress}(X) = \sum_{i<j} w_{ij}(\|X_i - X_j\| - d_{ij})^2
$$
Each pairwise term is denoted as:

$$
Q_{ij}(X) = w_{ij}(\|X_i - X_j\| - d_{ij})^2
$$



## Input Format

The input is provided as a MATLAB `.mat` file that stores matrices describing the graph structure and related parameters.Each file contains several fields under the Problem structure, including `title`, `A`, `name`, `id`, `date`, `author`, `ed`, and `kind`.

We only extract the adjacency matrix `A`:

| Key        | Type                                | Description                                                  |
| ---------- | ----------------------------------- | ------------------------------------------------------------ |
| **A**      | `n × n` numeric matrix              | The adjacency matrix of the graph. `A[i, j]` = 1 if there is an edge between nodes *i* and *j*, otherwise 0. |


All matrices are stored in standard MATLAB format and can be loaded in Python using:

```python
import scipy.io as sio
data = sio.loadmat(input_file)
graph = data['Problem']['A'][0][0]
```

The program should read these matrices, generate algorithms according to the stress minimization described above, and output the final 2D coordinates of each node.

## Output Format

The output are node coordinates. Each line corresponds to a single node, containing its index and the final two-dimensional coordinates (x, y), separated by spaces.  
The node indices must follow the same order as in the input matrices.

**Example:**
```
0  0.1325  0.8942
1  0.5278  0.3160
2  0.2411  0.7106
3  0.9182  0.1055
```

**Formatting rules:**

- Coordinates should be written as floating-point numbers, rounded to four decimal places.
- Nodes are 0-indexed (i.e., the first node is numbered 0).
- The output file should contain exactly n lines if the input graph has n nodes.
- The file should contain no extra headers, comments, or blank lines.
## Algorithm Guidelines (Robust & Scalable)

- No APSP on large graphs. Do not use all-pairs shortest paths (Floyd–Warshall, Johnson, or `shortest_path` in all-pairs mode).
- Distances (when needed).
  - Prefer unweighted BFS distances from sampled sources or local neighborhoods.
  - Do not build a full `N×N` distance matrix for large graphs.
- Initialization.**
  - Use spectral initialization (Laplacian eigenmaps): the 2nd and 3rd eigenvectors of the graph Laplacian as 2D coordinates.
  - If eigenvectors fail or are too expensive, fall back to fixed-seed random initialization.
- Optimization strategy.
  - Use edge-sampled SGD or mini-batch updates to approximate the stress gradient.
  - Optionally add lightweight repulsion using a small set of landmarks/subsampling (not full `N×N`).
  - Avoid triple nested loops and any dense `O(N^2)` / `O(N^3)` passes over all node pairs.
  - Prefer vectorized NumPy/SciPy ops and sparse matrices.
- Edge weights & graph handling.
  - Treat the adjacency as unweighted (0/1) unless explicitly specified.
  - Ensure non-negative weights; clamp negatives to `0`. Set diagonal to `0`. Do not create negative cycles.
- Determinism. Set a fixed random seed so runs are reproducible.
## References

1. Zheng, Jonathan X., Samraat Pawar, and Dan F. M. Goodman. 2019. “Graph Drawing by Stochastic Gradient Descent.” IEEE Transactions on Visualization and Computer Graphics 25(9): 2738–48.
