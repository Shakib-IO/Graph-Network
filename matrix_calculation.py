import numpy as np
import scipy.sparse as sp
import networkx as nx

"""Adjacency Matrix"""

"""
  Links:
    - https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix
    - 
"""

def normalize_symmetric_adj(mx):
    """Normalize sparse adjacency matrix with self-connection,
    https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/utils.py 
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx
  
def normalize(mx):
  """Row-normalize sparse matrix"""
  """An adjacency matrix can be row-normalised with A_row = D^−1 * A , or column normalised with A_col = A D^−1"""
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1.0).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv) # Construct a sparse matrix from diagonals.
  mx = r_mat_inv.dot(mx)
  return mx

def middle_normalize_adj(adj, alpha):
  """From this Paper: https://github.com/huangJC0429/Mid-GCN"""
  """Middle normalize adjacency matrix."""
  """Mid-GCN: (alpha * I - D^-1/2 * A * D^-1/2) * (I + D^-1/2 * A * D^-1/2) * XW"""

  rowsum = np.array(adj.sum(1))
  d_inv_sqrt = np.power(rowsum, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
  DAD = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
  return (alpha * sp.eye(adj.shape[0], adj.shape[1]) - DAD).dot(sp.eye(adj.shape[0], adj.shape[1]) + DAD)

edges = [[0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
nodes = sorted(set(node for edge in edges for node in edge))
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
A = nx.adjacency_matrix(G)
print(A.todense())

"""Apply the Formula"""
result = middle_normalize_adj(A, 0.1)
print(result.todense())
print("\n")
result_005 = middle_normalize_adj(A, 0.05)
print(result_005.todense())
