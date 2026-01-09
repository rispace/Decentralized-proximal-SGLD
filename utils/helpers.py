import numpy as np


def sample_uniform_l2_ball(dim, s, rng):
    z = rng.standard_normal(dim)
    z_norm = np.linalg.norm(z)
    if z_norm == 0:
        z = np.ones(dim)
        z_norm = np.linalg.norm(z)
    u = rng.random()
    r = s * (u ** (1.0 / dim))
    return r * (z / z_norm)


def sample_uniform_l1_ball(dim, s, rng):
    e = rng.exponential(scale=1.0, size=dim)
    v = e / e.sum()
    signs = rng.choice([-1, 1], size=dim)
    v = signs * v
    u = rng.random()
    r = s * (u ** (1.0 / dim))
    return r * v


def sample_uniform_lp_ball(dim, s, p, rng):
    if p == 2:
        return sample_uniform_l2_ball(dim, s, rng)
    elif p == 1:
        return sample_uniform_l1_ball(dim, s, rng)
    else:
        raise NotImplementedError("Only p=1 and p=2 are implemented.")


def priors(dim, s, p, N, rng):
    out = np.empty((N, dim))
    for i in range(N):
        out[i, :] = sample_uniform_lp_ball(dim, s, p, rng)
    return out


def project_onto_lp_ball(beta, p, s):
    """
    Project beta onto the Lp ball of radius s.
    
    Args: 
        beta: array-like, shape (dim,)
        p: int, either 1 or 2
        s: float, radius of the Lp ball
    """
    
    beta = np.asarray(beta, dtype=float)
    if p == 2:
        nrm = np.linalg.norm(beta, ord=2)
        if nrm <= s or nrm == 0:
            return beta
        return (s / nrm) * beta
    
    if p == 1:
        # L1 projection via sorting
        if np.linalg.norm(beta, ord=1) <= s:
            return beta
        
        u = np.sort(np.abs(beta))[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(
            u * np.arange(1, len(u) + 1) > (cssv - s)
        )[0][-1]
        theta = (cssv[rho] - s) / (rho + 1.0)
        return np.sign(beta) * np.maximum(np.abs(beta) - theta, 0.0)
    raise NotImplementedError("Only p=1 and p=2 are implemented.")