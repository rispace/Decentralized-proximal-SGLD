import numpy as np
from tqdm import tqdm
from utils.networks import Network
import utils.helpers as helpers


def gradient_Bayesian_LinearRegression(
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


def gradient_Bayesian_LogisticRegression(
    beta, X, y, gamma, batch_size, lp, s, rng
):
    idx = rng.integers(0, len(y), size=int(batch_size))
    Xi = X[idx]
    yi = y[idx]
    z = Xi @ beta 
    sig_z = 1.0 / (1.0 + np.exp(-z))
    grad = Xi.T @ (sig_z - yi)
    lp_norm = np.linalg.norm(beta, ord=lp)
    if lp_norm > s:
        grad += (
            beta - helpers.project_onto_lp_ball(beta, lp, s)
        ) / gamma
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
        self.X_local, self.y_local = self._split_data_among_agents(
            self.X, self.y, self.size_w, self.rng,
            stratify=(self.type == "logistic")
        )
    
    def _split_data_among_agents(
        self, X, y, size_w, rng, stratify=False
    ):
        """
        Returns lists: X_local[i], y_local[i]
        Each agent gets roughly n/size_w points
        stratify=True keeps class balance roughly equal per agent (binary y)
        """
        n = X.shape[0]
        idx = np.arange(n)
        
        if not stratify:
            rng.shuffle(idx)
            splits = np.array_split(idx, size_w)
        else:
            # simple stratified split for binary labels 0/1
            y = np.asarray(y)
            idx0 = idx[y == 0]
            idx1 = idx[y == 1]
            rng.shuffle(idx0)
            rng.shuffle(idx1)
            splits0 = np.array_split(idx0, size_w)
            splits1 = np.array_split(idx1, size_w)
            splits = [np.concatenate(
                (splits0[i], splits1[i])
                ) for i in range(size_w)]
            for s in splits:
                rng.shuffle(s)
        
        X_local = [X[s] for s in splits]
        y_local = [y[s] for s in splits]
        
        return X_local, y_local
    
    def _dpsgld_step(self, B):
        """
        One step of DPSGLD update
        B (N, size_w, dim)
        """
        for n in range(self.N):
            for i in range(self.size_w):
                if self.type == "linear":
                    grad = gradient_Bayesian_LinearRegression(
                        B[n, i], self.X_local[i], self.y_local[i],
                        self.gamma, self.batch_size,
                        self.lp, self.s, self.sigma, 
                        self.rng
                    )
                    n_total = self.y.shape[0]
                    n_i = self.y_local[i].shape[0]
                    grad = (n_total / n_i) * grad
                elif self.type == "logistic":
                    grad = gradient_Bayesian_LogisticRegression(
                        B[n, i], self.X_local[i], self.y_local[i],
                        self.gamma, self.batch_size,
                        self.lp, self.s, self.rng
                    )
                    n_total = self.y.shape[0]
                    n_i = self.y_local[i].shape[0]
                    grad = (n_total / n_i) * grad
                else:
                    raise NotImplementedError(
                        "Only linear regression is implemented."
                    )
                temp = np.zeros(shape=(self.dim))
                for j in range(self.size_w):
                    temp = temp + self.W[i, j] * B[n, j]
                noise = self.rng.standard_normal(self.dim)
                B[n, i] = (
                    temp - self.eta * (grad)
                    + np.sqrt(2.0 * self.eta) * noise
                )
        return B
    
    def _mysgld_step(self, B):
        """
        One step of MySGLD update
        B (N, dim)
        """
        for n in range(self.N):
            if self.type == "linear":
                grad = gradient_Bayesian_LinearRegression(
                    B[n], self.X, self.y,
                    self.gamma, self.batch_size,
                    self.lp, self.s, self.sigma,
                    self.rng
                )
            elif self.type == "logistic":
                grad = gradient_Bayesian_LogisticRegression(
                    B[n], self.X, self.y,
                    self.gamma, self.batch_size,
                    self.lp, self.s, self.rng
                )
            else:
                raise NotImplementedError(
                    "Only linear regression is implemented."
                )
            noise = self.rng.standard_normal(self.dim)
            B[n] = (
                B[n] - self.eta * (grad)
                + np.sqrt(2.0 * self.eta) * noise
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
            Betas = helpers.priors(
                self.dim, self.s, self.lp, self.N * self.size_w,
                rng=self.rng
            ).reshape(self.N, self.size_w, self.dim)
            history_all = []
            B_mean_all = []
            for t in range(1):
                history = np.empty((self.size_w, self.dim, self.N))
                B_mean = np.empty((self.dim, self.N))
                for i in range(self.N):
                    history[:, :, i] = Betas[i, :, :]
                for j in range(self.dim):
                    B_mean[j, :] = np.mean(history[:, j, :], axis=0)
                history_all.append(history)
                B_mean_all.append(B_mean)
            # Update parameters using DPSGLD
            for k in tqdm(range(self.n_iteration)):
                Betas = self._dpsgld_step(Betas)
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