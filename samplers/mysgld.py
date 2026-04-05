import numpy as np
from tqdm import tqdm
import utils.helpers as helpers


class BayesianRegressionMYSGLD:
    def __init__(
        self, X: np.ndarray, y: np.ndarray,
        n_samples: int = 100, batch: int = 100,
        eta: float = 5e-4, gamma: float = 1e-2,
        sigma: float = 0.5, lp: int = 2,
        s: float = 0.8, n_iters: int = 200,
        seed: int = None, type: str = None
    ):
        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.batch = batch
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.lp = lp
        self.s = s
        self.n_iters = n_iters
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.type = type
        
        if self.type is None:
            raise ValueError(
                "Model type (linear or logistic) must be specified."
            )
    
    def _mysgld_update(self, beta):
        for j in range(self.n_samples):
            if self.type == "linear":
                grad = helpers.grad_BayesianLinearRegression(
                    beta[:, j], self.X, self.y, self.gamma,
                    self.batch, self.lp, self.s, self.sigma, self.rng
                )
            elif self.type == "logistic":
                grad = helpers.grad_BayesianLogisticRegression(
                    beta[:, j], self.X, self.y, self.gamma,
                    self.batch, self.lp, self.s, self.rng
                )
            else:
                raise NotImplementedError(
                    "Model type must be either 'linear' or 'logistic'."
                )
            noise = self.rng.standard_normal(self.X.shape[1])
            beta[:, j] -= self.eta * grad + np.sqrt(2 * self.eta) * noise
        return
    
    def sample_parameters(
        self, iterations: int = None
    ):
        if iterations is not None:
            self.n_iters = iterations
            
        beta = helpers.priors(
            dim=self.X.shape[1], s=self.s, lp=self.lp,
            N=self.n_samples, rng=self.rng
        ).reshape(self.X.shape[1], self.n_samples)  # shape (dim, n_samples)
        
        chain = np.empty((self.n_iters+1, self.X.shape[1], self.n_samples))
        chain[0, :, :] = beta
        
        for t in tqdm(range(self.n_iters)):
            self._mysgld_update(beta)
            chain[t+1, :, :] = beta
        
        return np.array(chain)


class MYSGLD1D:
    """
    Moreau–Yosida (indicator) based centralized SGLD in 1D.
    Targets pi^gamma(x) ∝ exp( -U(x) - (1/(2gamma))||x - proj_K(x)||^2 ).
    """
    def __init__(
        self, eta, n_samples, n_iters=200,
        sigma_grad=0.5, gamma=1e-2,
        mu=1.0, beta=0.5, b=1.0,
        R=1.0, seed=None
    ):
        self.eta = float(eta)
        self.n_samples = int(n_samples)
        self.n_iters = int(n_iters)
        self.sigma_grad = float(sigma_grad)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.beta = float(beta)
        self.b = float(b)
        self.R = float(R)

        self.rng = np.random.default_rng(seed)

    def projK(self, x):
        return np.clip(x, -self.R, self.R)

    def stochastic_gradient(self, x):
        grad = self.mu * x + self.beta * x**3 - self.b
        noise = self.sigma_grad * self.rng.standard_normal()
        return grad + noise

    def step(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)  # shape (N,)
        x_new = np.empty_like(x)

        for n in range(self.n_samples):
            grad_n = self.stochastic_gradient(x[n]) 
            prox_n = (x[n] - self.projK(x[n])) / self.gamma  
            noise_n = np.sqrt(2.0 * self.eta) * self.rng.standard_normal()
            x_new[n] = x[n] - self.eta * (grad_n + prox_n) + noise_n

        return x_new

    def sample(self, x0=None):
        x = (
            self.rng.standard_normal(self.n_samples)
            if x0 is None else np.array(x0, dtype=float).reshape(-1)
        )

        chain = np.empty((self.n_iters + 1, self.n_samples))
        chain[0] = x
        for k in tqdm(range(self.n_iters)):
            x = self.step(x)
            chain[k + 1] = x

        return np.array(chain)