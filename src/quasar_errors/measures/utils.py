__all__ = [
    'mirror_dist',
    'mirror_and_rebin',
]

from numpy import append, searchsorted, invert, full, zeros, float64
from numba import njit
from quasar_typing.numpy import FloatVector
from quasar_utils.binning import alpha_matrix_sparse

@njit(fastmath=True)
def mirror_dist(
    cen: float,
    x: FloatVector,
    y: FloatVector,
    dx: FloatVector,
) -> tuple[FloatVector, FloatVector, FloatVector, FloatVector, FloatVector]:
    """
    ** NUMBA OPTIMISED FUNCTION **
    """
    edges = append(
        x - dx / 2,
        x[-1] + dx[-1] / 2,
    )
    idx_cut = searchsorted(edges, cen, side='right') - 1
    val_cut = y[idx_cut]
    lb, ub = edges[idx_cut:idx_cut+2]

    # LHS
    mask_l = (edges < cen)
    edges_l = append(
        0,
        cen - edges[mask_l][::-1],
    )
    vals_l = append(
        val_cut * (cen - lb) / (ub - lb),
        y[:idx_cut][::-1],
    )

    mask_r = invert(mask_l)
    edges_r = append(
        0, 
        edges[mask_r] - cen,
    )
    vals_r = append(
        val_cut * (ub - cen) / (ub - lb),
        y[idx_cut+1:],
    )

    new_edges = append(
        edges_l,
        edges_r[1:],
    )
    new_edges.sort()

    return edges_l, edges_r, vals_l, vals_r, new_edges

def mirror_and_rebin(
    cen: float,
    x: FloatVector,
    y: FloatVector,
    dx: FloatVector,
) -> tuple[FloatVector, FloatVector]:

    edges_l, edges_r, vals_l, vals_r, new_edges = mirror_dist(
        cen, 
        x, 
        y, 
        full(x.size, dx) if isinstance(dx, float) else dx,
    )
    
    new_vals = zeros(new_edges.size - 1, dtype=float64)

    a_l = alpha_matrix_sparse.__wrapped__(edges_l, new_edges)
    if a_l.sum() > 0:
        new_vals += a_l.dot(vals_l) / a_l.sum()
    
    a_r = alpha_matrix_sparse.__wrapped__(edges_r, new_edges)
    if a_r.sum() > 0:
        new_vals += a_r.dot(vals_r) / a_r.sum()

    return new_edges, new_vals