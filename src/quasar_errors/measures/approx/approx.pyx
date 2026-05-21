from libc.math cimport sqrt, fabs

cdef inline int binary_search_c(
    double x,
    double[:] xp,
) nogil:
    """Find index i where xp[i] <= x < xp[i+1]."""
    cdef int left = 0
    cdef int right = len(xp) - 2  # -2 because we need xp[i+1]
    cdef int mid
    
    while left < right:
        mid = (left + right) >> 1  # Faster than / 2
        if xp[mid] < x:
            left = mid + 1
        else:
            right = mid
    
    return left

cdef inline double interp_c(
    double x,
    double[:] xp,
    double[:] yp,
) nogil:
    cdef int i = binary_search_c(x, xp)
    
    # Clamp to valid range
    if i >= len(xp) - 1:
        i = len(xp) - 2
    
    # Linear interpolation
    if xp[i+1] == xp[i]:
        return yp[i]
    
    cdef double frac = (x - xp[i]) / (xp[i+1] - xp[i])
    return yp[i] + frac * (yp[i+1] - yp[i])

cdef inline (double, double) u_yinterp_c(
    double x, 
    double[:] xp, 
    double[:] yp, 
    double[:] dyp,
) noexcept nogil:
    cdef double dx = xp[1] - xp[0]
    cdef double f0 = (xp[1] - x) / dx
    cdef double f1 = (x - xp[0]) / dx
    cdef double dy_term0 = f0 * dyp[0]
    cdef double dy_term1 = f1 * dyp[1]
    
    return (
        f0 * yp[0] + f1 * yp[1],
        sqrt(dy_term0 * dy_term0 + dy_term1 * dy_term1)
    )

def u_yinterp(double x, double[:] xp, double[:] yp, double[:] dyp):
    return u_yinterp_c(x, xp, yp, dyp)

cdef inline (double, double) u_xinterp_c(
    double y,
    double[:] xp,
    double[:] yp,
    double[:] dyp,
) noexcept nogil:    
    cdef double dx = xp[1] - xp[0]
    cdef double dy = yp[1] - yp[0]
    cdef double f0 = (yp[1] - y) / dy
    cdef double f1 = (y - yp[0]) / dy
    
    cdef double dy_term0 = f0 * dyp[0]
    cdef double dy_term1 = f1 * dyp[1]
    cdef double scale = fabs(dx) / fabs(dy)
    
    return (
        f0 * xp[0] + f1 * xp[1],
        scale * sqrt(dy_term0 * dy_term0 + dy_term1 * dy_term1)
    )

def u_xinterp(double y, double[:] xp, double[:] yp, double[:] dyp):
    return u_xinterp_c(y, xp, yp, dyp)

cdef (double, double) get_intercept_c(
    double[:] x,
    double[:] y,
    double[:] dy,
    double y_crit,
    int strategy,
):
    """
    Pure C implementation of get_intercept (cdef).
    
    Finds the x-value where the curve crosses y_crit (upward crossings only), 
    with uncertainty propagation.
    
    strategy: 0=narrowest, 1=average, 2=widest
    """
    cdef int i, n, idx_r, n_crossings, first_crossing, last_crossing
    cdef double left_x, left_dy, right_x, right_dy, avg_x, avg_dy
    
    n = len(x)
    first_crossing = -1
    last_crossing = -1
    n_crossings = 0
    
    # Find all UPWARD crossings (y[i] < y_crit and y[i+1] >= y_crit)
    for i in range(n - 1):
        if (y[i] < y_crit) and (y[i+1] >= y_crit):
            if first_crossing == -1:
                first_crossing = i
            last_crossing = i
            n_crossings += 1
    
    # Handle edge cases
    if n_crossings == 0:
        if y[0] >= y_crit:
            return (x[0], dy[0])
        else:
            return (x[n-1], dy[n-1])
    
    # Single crossing: use it
    if n_crossings == 1:
        idx_r = first_crossing
        return u_xinterp_c(
            y_crit,
            x[idx_r:idx_r+2],
            y[idx_r:idx_r+2],
            dy[idx_r:idx_r+2],
        )
    
    if strategy == 1:  # average
        # First crossing
        idx_r = first_crossing
        left_x, left_dy = u_xinterp_c(
            y_crit,
            x[idx_r:idx_r+2],
            y[idx_r:idx_r+2],
            dy[idx_r:idx_r+2],
        )
        
        # Last crossing
        idx_r = last_crossing
        right_x, right_dy = u_xinterp_c(
            y_crit,
            x[idx_r:idx_r+2],
            y[idx_r:idx_r+2],
            dy[idx_r:idx_r+2],
        )
        
        avg_x = (left_x + right_x) / 2.0
        avg_dy = sqrt(left_dy*left_dy + right_dy*right_dy) / 2.0
        return (avg_x, avg_dy)
    elif strategy == 2:  # widest
        idx_r = first_crossing
    else:  # narrowest
        idx_r = last_crossing
    
    return u_xinterp_c(
        y_crit,
        x[idx_r:idx_r+2],
        y[idx_r:idx_r+2],
        dy[idx_r:idx_r+2],
    )

def get_intercept(
    double[:] x,
    double[:] y,
    double[:] dy,
    double y_crit,
    int strategy,
):
    return get_intercept_c(x, y, dy, y_crit, strategy)

cdef (double, double) get_width_at_height_c(
    double[:] x,
    double[:] y,
    double[:] dy,
    double x_max,
    double y_crit,
    int strategy,
):
    """
    Pure C implementation of get_width_at_height (cdef).
    
    Measures the width of the curve at height y_crit, within x <= x_max.
    
    strategy: 0=narrowest, 1=average, 2=widest
    """
    cdef int n, i, max_idx, right_start
    cdef double left_x, left_dy, right_x, right_dy
    cdef double width, uncertainty
    cdef double[:] x_left_part
    cdef double[:] y_left_part
    cdef double[:] dy_left_part
    cdef double[:] x_right_part
    cdef double[:] y_right_part
    cdef double[:] dy_right_part
    
    n = len(x)
    
    # Find the index where x crosses x_max (last index where x <= x_max)
    max_idx = -1
    for i in range(n):
        if x[i] <= x_max:
            max_idx = i
        else:
            break
    
    # Get left part of spectrum (x <= x_max)
    left_x = x[0]
    left_dy = 0.0
    if max_idx >= 0:
        x_left_part = x[:max_idx+1]
        y_left_part = y[:max_idx+1]
        dy_left_part = dy[:max_idx+1]
        
        left_x, left_dy = get_intercept_c(
            x_left_part, y_left_part, dy_left_part, y_crit, strategy
        )
    
    # Get right part of spectrum (x > x_max), reversed
    right_x = x[n-1]
    right_dy = 0.0
    right_start = max_idx + 1
    if right_start < n:
        # Create reversed views of right part
        x_right_part = x[right_start:n][::-1]
        y_right_part = y[right_start:n][::-1]
        dy_right_part = dy[right_start:n][::-1]
        
        right_x, right_dy = get_intercept_c(
            x_right_part, y_right_part, dy_right_part, y_crit, strategy
        )
    
    width = right_x - left_x
    uncertainty = sqrt(left_dy*left_dy + right_dy*right_dy)
    
    return (width, uncertainty)


def get_width_at_height(
    double[:] x,
    double[:] y,
    double[:] dy,
    double x_max,
    double y_crit,
    int strategy,
):
    """
    Wrapper for get_width_at_height_c.
    
    Measures the width of the curve at height y_crit, within x <= x_max.
    """
    return get_width_at_height_c(x, y, dy, x_max, y_crit, strategy)

cdef int get_quantile_c(
    double[:] q,
    double[:] x,
    double[:] y,
    double[:] edges,
    double[:] cdf,
    double[:] out,
) nogil:
    """
    Inplace function: out
    """
    cdef int i, j, n_pix, n_solutions
    cdef double cdf_sum
    
    n_pix = len(x)
    n_solutions = len(q)

    cdf_sum = 0.0
    cdf[0] = 0.0
    for i in range(n_pix):
        cdf_sum += y[i]
        cdf[i + 1] = cdf_sum

    if cdf_sum > 0:
        for i in range(n_pix + 1):
            cdf[i] /= cdf_sum

    for j in range(n_solutions):
        out[j] = interp_c(q[j], cdf, edges)

    return 0

def get_quantile(
    double[:] q,
    double[:] x,
    double[:] y,
    double[:] edges,
    double[:] cdf,
    double[:] out,
):  
    return get_quantile_c(q, x, y, edges, cdf, out)

cdef (double, double) get_peak_c(
    double[:] x,
    double[:] y,
    double[:] edges,
) noexcept nogil:
    cdef int i, n_pix, idx_max
    cdef double y_max, x_max, xl, xr, yl, yr

    n_pix = len(x)
    y_max = y[0]
    idx_max = 0
    for i in range(1, n_pix):
        if y[i] > y_max:
            y_max = y[i]
            idx_max = i

    if (idx_max == 0) or (idx_max == n_pix - 1):
        x_max = x[idx_max]
    else:
        xl = edges[idx_max]
        xr = edges[idx_max + 1]
        yl = y[idx_max - 1]
        yr = y[idx_max + 1]
        x_max = xl + (xr - xl) * ((y_max - yl) / (2 * y_max - yl - yr))

    return (x_max, y_max)

def get_peak(double[:] x, double[:] y, double[:] edges):
    return get_peak_c(x, y, edges)