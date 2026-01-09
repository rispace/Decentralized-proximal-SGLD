import numpy as np
from tqdm import tqdm
from utils.networks import Network
import utils.helpers as helpers


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


def stochastic_gradient_linear_regression(
    beta, X, y, batch_size, sigma2, rng
):
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    
    idx = rng.integers(low=0, high=n, size=batch_size)
    Xb = X[idx]
    yb = y[idx]
    scale = n / batch_size
    r = Xb @ beta - yb
    grad = scale * (Xb.T @ r) / sigma2
    return grad    
    

class BayesianRegression:
    def __init__(
        self, X: np.ndarray = None, y: np.ndarray = None,
        N: int = 100, batch_size: int = 50, eta: float = 5e-4,
        gamma: float = 1e-2, lp: int = 1, s: float = 1.5,
        n_iteration: int = 10000, size_w: int = 8,
        sigma: float = 1.0, net: str = "cn",
        seed: int = None, type: str = "linear",
    ):
        self.X = X
        self.y = y
        self.N = N
        self.batch_size = batch_size
        self.eta = eta
        self.gamma = gamma
        self.lp = lp
        self.s = s
        self.n_iteration = n_iteration
        self.size_w = size_w
        self.sigma = sigma
        self.net = net
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.type = type
        
        if self.type is None:
            raise ValueError(
                "Model type (linear or logistic) must be specified."
            )
        
        nets = Network(
            size_w=self.size_w,
            seed=self.seed if self.seed is not None else 42
        )
        if self.net == "fcn":
            self.W = nets.fully_connected_network()
        elif self.net == "cn":
            self.W = nets.circular_network()
        elif self.net == "sn":
            self.W = nets.star_network()
        elif self.net == "fdn":
            self.W = nets.disconnected_network()
        else:
            raise ValueError("Unknown network type.")
        
        if (self.X is None) or (self.y is None):
            raise ValueError("Data X and y must be provided.")
        
        self.dim = self.X.shape[1]
    
    def _dpsgld_step(self, B):
        """
        One step of DPSGLD update
        B (N, size_w, dim)
        """
        for n in range(self.N):
            for i in range(self.size_w):
                if self.type == "linear":
                    grad = stochastic_gradient_linear_regression(
                        B[n, i], self.X, self.y,
                        self.batch_size, sigma2=self.sigma**2,
                        rng=self.rng
                    )
                    prox_grad = (
                        B[n, i] - project_onto_lp_ball(
                            B[n, i], self.lp, self.s
                        )
                    ) / self.gamma
                    temp = np.zeros(shape=(self.dim))
                    for j in range(len(B[n])):
                        temp = temp + self.W[i, j] * B[n, j]
                    noise = self.rng.standard_normal(
                        self.dim
                    )
                    B[n, i] = (
                        temp - self.eta * (grad + prox_grad)
                        + np.sqrt(2.0 * self.eta) * noise
                    )
                else:
                    raise NotImplementedError(
                        "Only linear regression is implemented."
                    )
        return B
    
    def _mysgld_step(self, B):
        """
        One step of MySGLD update
        B (N, dim)
        """
        for n in range(self.N):
            if self.type == "linear":
                grad = stochastic_gradient_linear_regression(
                    B[n], self.X, self.y,
                    self.batch_size, sigma2=self.sigma**2,
                    rng=self.rng
                )
                prox_grad = (
                    B[n] - project_onto_lp_ball(
                        B[n], self.lp, self.s
                    )
                ) / self.gamma
                noise = self.rng.standard_normal(
                    self.dim
                )
                B[n] = (
                    B[n] - self.eta * (grad + prox_grad)
                    + np.sqrt(2.0 * self.eta) * noise
                )
            else:
                raise NotImplementedError(
                    "Only linear regression is implemented."
                )
        return B
    
    def sample_parameters(
        self, method=None, iterations: int = None
    ):
        if method is None:
            raise ValueError(
                "Sampling method must be specified:"
                "E.g. 'dpsgld', 'mysgld', etc"
            )
        
        if iterations is not None:
            self.n_iteration = iterations
        
        if method == "dpsgld":
            # Initialize parameters from prior
            B = priors(
                self.dim, self.s, self.lp, self.N * self.size_w,
                rng=self.rng
            ).reshape(self.N, self.size_w, self.dim)
            history_all = []
            B_mean_all = []
            for t in range(1):
                history = np.empty((self.size_w, self.dim, self.N))
                B_mean = np.empty((self.dim, self.N))
                for i in range(self.N):
                    history[:, :, i] = B[i, :, :]
                for j in range(self.dim):
                    B_mean[j, :] = np.mean(history[:, j, :], axis=0)
                history_all.append(history)
                B_mean_all.append(B_mean)
            # Update parameters using DPSGLD
            for k in tqdm(range(self.n_iteration)):
                B = self._dpsgld_step(B)
                history = np.empty((self.size_w, self.dim, self.N))
                B_mean = np.empty((self.dim, self.N))
                for i in range(self.N):
                    history[:, :, i] = B[i, :, :]
                for j in range(self.dim):
                    B_mean[j, :] = np.mean(history[:, j, :], axis=0)
                history_all.append(history)
                B_mean_all.append(B_mean)
            
            return np.array(history_all), np.array(B_mean_all)
        elif method == "mysgld":
            # Initialize 
            B = priors(
                self.dim, self.s, self.lp, self.N,
                rng=self.rng
            ).reshape(self.N, self.dim)
            
            # Update parameters using MySGLD
            chain = np.empty((self.n_iteration+1, self.dim, self.N))
            chain[0, :] = B.T
            for k in tqdm(range(self.n_iteration)):
                B = self._mysgld_step(B)
                chain[k + 1, :] = B.T
            
            return np.array(chain)
        else:
            raise NotImplementedError(
                "Only 'dpsgld' and 'mysgld' methods are implemented."
            )