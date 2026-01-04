import numpy as np
from tqdm import tqdm
from utils.networks import Network

class DPSGLD1D:
    """
    Decentralized Proximal SGLD in one dimension
    
    Args: 
        size_w (int): Size of the doubly stochastic matrix aka mixing matrix W
        N (int): Number of samples to generate
        n_steps (int): Number of iteration
        eta (float): step size or learning rate
        Potential U(x) Args:
            mu (float): Hyperparameter; default 1.0
            beta (float): Hyperparameter; default 0.5
            b (float): Hyperparameter; default 1.0

        sigma_grad (float): Standard Deviation for the gradient noise
        gamma (float): Proximal Regularizer
        net (str): Network type; Takes only "fcn", "cn", "sn", or "fdn"
        R (float): Boundary of the convex domain
    """
    def __init__(self, size_w, N, eta, n_steps, dim=1,
                 mu=1.0, beta=0.5, b=1.0, sigma_grad=0.5,
                 gamma=1e-2, net='cn', R=1.0, seed=None):
        self.size_w=size_w
        self.N=N
        self.eta=eta
        self.n_steps=n_steps
        self.mu=mu
        self.beta=beta
        self.b=b
        self.sigma_grad=sigma_grad
        self.gamma=gamma
        self.net=net
        self.R=R
        self.dim=dim
        self.seed=seed


        # single RNG for the whole class
        self.rng = np.random.default_rng(seed)

        nets = Network(size_w=self.size_w, seed=self.seed if self.seed is not None else 42)
        if self.net == 'fcn':
            self.W = nets.fully_connected_network()
        elif self.net == 'cn':
            self.W = nets.circular_network()
        elif self.net == 'sn':
            self.W = nets.star_network()
        elif self.net == 'fdn':
            self.W = nets.disconnected_network()
        else:
            raise ValueError(f'Network must be one of "fcn", "cn", "sn", or "fdn", but got {self.net}')

    def project_to_K(self, x):
        return np.clip(x, -self.R, self.R)

    def stochastic_gradient(self, x):
        grad_fi = self.mu * x + self.beta * x**3 - self.b
        noise = self.sigma_grad * self.rng.standard_normal()
        return grad_fi + noise

    
    def step(self, X):
        for n in range(self.N):
            for i in range(self.size_w):
                grad = self.stochastic_gradient(X[n,i])
                prox_grad = (X[n, i] - self.project_to_K(X[n, i])) / self.gamma
                Langevin_noise = np.sqrt(2.0 * self.eta) * self.rng.standard_normal()
                temp = np.zeros(self.dim)
                for j in range(len(X[n])):
                    temp += self.W[i,j] * X[n, j]
                X[n, i] = (
                    temp - self.eta * (grad + prox_grad) + Langevin_noise
                )
        return X
    
    def sample(self):
        # Initialization
        X = np.random.normal(loc=0, scale=1, size=(self.N, self.size_w, self.dim))
        
        history_all = []
        X_mean_all = []
        
        for t in range(1):
            history = np.empty((self.size_w, self.dim, self.N))
            X_mean = np.empty((self.dim, self.N))
            for i in range(self.N):
                history[:, :, i] = X[i, :, :]
            X_mean[0,:] = np.mean(history[:, 0, :], axis=0)
            history_all.append(history)
            X_mean_all.append(X_mean)
        # Update  
        for k in tqdm(range(self.n_steps)):
            X = self.step(X)
            history = np.empty((self.size_w, self.dim, self.N))
            X_mean = np.empty((self.dim, self.N))
            
            for i in range(self.N):
                history[:, :, i] = X[i, :, :]
            X_mean[0,:] = np.mean(history[:, 0, :], axis=0)
            history_all.append(history)
            X_mean_all.append(X_mean)
        
        return np.array(history_all), np.array(X_mean_all)


class MYSGLD1D:
    """
    Moreau–Yosida (indicator) based centralized SGLD in 1D.
    Targets pi^gamma(x) ∝ exp( -U(x) - (1/(2gamma))||x - proj_K(x)||^2 ).
    """
    def __init__(
        self, eta, N, n_steps=200,
        sigma_grad=0.5, gamma=1e-2,
        mu=1.0, beta=0.5, b=1.0,
        R=1.0, seed=None
    ):
        self.eta = float(eta)
        self.N = int(N)
        self.n_steps = int(n_steps)
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

        for n in range(self.N):
            grad_n = self.stochastic_gradient(x[n])          # scalar grad + scalar grad-noise
            prox_n = (x[n] - self.projK(x[n])) / self.gamma  # scalar prox
            noise_n = np.sqrt(2.0 * self.eta) * self.rng.standard_normal()
            x_new[n] = x[n] - self.eta * (grad_n + prox_n) + noise_n

        return x_new

    def sample(self, x0=None):
        x = self.rng.standard_normal(self.N) if x0 is None else np.array(x0, dtype=float).reshape(-1)

        chain = np.empty((self.n_steps +1, self.N))
        chain[0] = x
        for k in tqdm(range(self.n_steps)):
            x = self.step(x)
            chain[k + 1] = x

        return np.array(chain)