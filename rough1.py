import numpy as np
from tqdm import tqdm
from utils.networks import Network
import utils.helpers as helpers


def gradient_BLR(
    beta, X, y, gamma, batch_size, lp, s, sigma, rng
):
    dim = X.shape[1]
    randomidx = rng.integers(0, len(y), size=int(batch_size))
    grad_f = np.zeros(shape=(dim))
    for i in randomidx:
        grad_f -= ((y[i] - beta @ X[i]) * X[i]) / (batch_size * sigma**2)
    lp_norm = np.linalg.norm(beta, ord=lp)
    if lp_norm > s:
        grad_f += (
            beta - helpers.project_onto_lp_ball(beta, lp, s)
        ) / gamma
    return grad_f             


def gradient_BLogR(
    beta, X, y, gamma, batch_size, lp, s, rng
):
    dim = X.shape[1]
    idx = rng.integers(0, len(y), size=int(batch_size))
    grad_f = np.zeros(dim)
    for i in idx:
        p = helpers.sigmoid(beta @ X[i])
        grad_f += (
            (p - y[i]) * X[i]
        ) / batch_size
    lp_norm = np.linalg.norm(beta, ord=lp)
    if lp_norm > s:
        grad_f += (
            beta - helpers.project_onto_lp_ball(beta, lp, s)
        ) / gamma
    return grad_f


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
    
    def _dpsgld_step(self, B, eta):
        """
        One step of DPSGLD update
        B (N, size_w, dim)
        """
        for n in range(self.N):
            for i in range(self.size_w):
                if self.type == "linear":
                    grad = gradient_BLR(
                        B[n, i], self.X, self.y,
                        self.gamma, self.batch_size,
                        self.lp, self.s, self.sigma, 
                        self.rng
                    )
                elif self.type == "logistic":
                    grad = gradient_BLogR(
                        B[n, i], self.X, self.y,
                        self.gamma, self.batch_size,
                        self.lp, self.s,
                        self.rng
                    )
                else:
                    raise NotImplementedError(
                        "Only linear and logistic regressions are implemented."
                    )
                temp = np.zeros(shape=(self.dim))
                for j in range(self.size_w):
                    temp = temp + self.W[i, j] * B[n, j]
                noise = self.rng.standard_normal(
                    self.dim
                )
                B[n, i] = (
                    temp - eta * (grad)
                    + np.sqrt(2.0 * eta) * noise
                )
        return B
    
    def _mysgld_step(self, B, eta):
        """
        One step of MySGLD update
        B (N, dim)
        """
        for n in range(self.N):
            if self.type == "linear":
                grad = gradient_BLR(
                    B[n], self.X, self.y,
                    self.gamma, self.batch_size,
                    self.lp, self.s, self.sigma,
                    self.rng
                )
            elif self.type == "logistic":
                grad = gradient_BLogR(
                    B[n], self.X, self.y,
                    self.gamma, self.batch_size,
                    self.lp, self.s,
                    self.rng
                )
            else:
                raise NotImplementedError(
                    "Only linear and logistic regressions are implemented."
                )
            noise = self.rng.standard_normal(
                self.dim
            )
            B[n] = (
                B[n] - eta * (grad)
                + np.sqrt(2.0 * eta) * noise
            )
        return B
    
    def sample_parameters(
        self, method=None, iterations: int = None,
        burn_in: int = 0, eta_decay: bool = True,
        eta_decay_fractions=(0.2, 0.4, 0.6, 0.8),
        eta_decay_rate: float = 0.5
    ):
        if method is None:
            raise ValueError(
                "Sampling method must be specified:"
                "E.g. 'dpsgld', 'mysgld', etc"
            )
        
        if iterations is not None:
            self.n_iteration = iterations
        
        T = int(self.n_iteration)
        if T <= 0:
            raise ValueError("Number of iterations must be positive.")
        if burn_in < 0 or burn_in >= T:
            raise ValueError("Invalid burn-in period.")
        
        eta0 = float(self.eta)
        
        fractions = sorted(float(f) for f in eta_decay_fractions)
        milestones = []
        for f in fractions:
            if not (0.0 < f < 1.0):
                raise ValueError(
                    "Eta decay fractions must be in (0, 1)."
                )
            m = int(np.floor(f * T))
            m = max(1, min(T-1, m))
            milestones.append(m)
        milestones = sorted(set(milestones))
        
        def get_eta(k):
            if not eta_decay or len(milestones) == 0:
                return eta0
            completed = k + 1
            drops = 0
            for m in milestones:
                if completed >= m:
                    drops += 1
                else:
                    break
            return eta0 * (eta_decay_rate ** drops)
        
        prev_eta = None
        
        if method == "dpsgld":
            # Initialize parameters from prior
            Betas = helpers.priors(
                self.dim, self.s, self.lp, self.N * self.size_w,
                rng=self.rng
            ).reshape(self.N, self.size_w, self.dim)
            history_all = []
            B_mean_all = []
            
            # Update parameters using DPSGLD
            for k in tqdm(range(self.n_iteration)):
                eta = get_eta(k)
                if prev_eta is None or eta != prev_eta:
                    print(f"Iteration {k+1}/{self.n_iteration}, eta={eta}")
                    prev_eta = eta
                Betas = self._dpsgld_step(Betas, eta)
                
                if k < burn_in:
                    continue
                
                history = np.empty((self.size_w, self.dim, self.N))
                B_mean = np.empty((self.dim, self.N))
                for i in range(self.N):
                    history[:, :, i] = Betas[i, :, :]
                for j in range(self.dim):
                    B_mean[j, :] = np.mean(history[:, j, :], axis=0)
                history_all.append(history)
                B_mean_all.append(B_mean)
            
            return np.array(history_all), np.array(B_mean_all)
        elif method == "mysgld":
            # Initialize 
            B = helpers.priors(
                self.dim, self.s, self.lp, self.N,
                rng=self.rng
            ).reshape(self.N, self.dim)
            
            # Update parameters using MySGLD
            chain = []
            for k in tqdm(range(self.n_iteration)):
                eta = get_eta(k)
                if prev_eta is None or eta != prev_eta:
                    print(f"Iteration {k+1}/{self.n_iteration}, eta={eta}")
                    prev_eta = eta
                B = self._mysgld_step(B, eta)
                
                if k < burn_in:
                    continue
                chain.append(B.T.copy())
            
            return np.array(chain)
        else:
            raise NotImplementedError(
                "Only 'dpsgld' and 'mysgld' methods are implemented."
            )