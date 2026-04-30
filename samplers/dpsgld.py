import numpy as np
from tqdm import tqdm
from utils.networks import Network
import utils.helpers as helpers


class BayesianRegressionDPSGLD:
    """
    Decentralized Proximal SGLD for Bayesian Linear and Logistic Regression
    
    Args:
        X (np.ndarray): Input data of shape (N, d)
        y (np.ndarray): Target labels of shape (N,)
        n_samples (int): Number of samples to generate
        batch (int): Batch size for stochastic gradient estimation
        eta (float): Step size or learning rate
        gamma (float): Proximal regularizer
        lp (int): Norm degree for the prior distribution 
                  (e.g., 1 for Laplace, 2 for Gaussian)
        s (float): Scale parameter for the prior distribution
        n_iters (int): Number of iterations for sampling
        n_agents (int): Number of agents in the decentralized network
        sigma (float): Standard deviation for the gradient noise
        net (str): Network type; Takes only "fcn", "cn", "sn", or "fdn"
        seed (int): Random seed for reproducibility
        type (str): Model type; Takes only "linear" or "logistic"
    
        Note: The input data X and labels y are expected to be partitioned 
        across agents, meaning that each agent has access to a subset of the
        data. The sampling process will be performed in a decentralized manner,
        where each agent updates its parameters based on its local data and
        communicates with neighboring agents according to the specified
        network topology.
    
    Returns:
        history_all_agents (np.ndarray): 
                A list of parameter samples for all agents across iterations
        beta_mean_all_agents (np.ndarray): 
            A list of mean parameter estimates across agents for each iteration
    """
    
    def __init__(
        self, X: np.ndarray, y: np.ndarray,
        n_samples: int = 100, batch: int = 10, eta: float = 5e-4,
        gamma: float = 1e-2, lp: int = 1, s: float = 0.8,
        n_iters: int = 200, n_agents: int = 10,
        sigma: float = 1.0, net: str = "cn",
        seed: int = None, type: str = None
    ):
        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.batch = batch
        self.eta = eta
        self.gamma = gamma
        self.lp = lp
        self.s = s
        self.n_iters = n_iters
        self.n_agents = n_agents
        self.sigma = sigma
        self.net = net
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.type = type
        self.dim = X.shape[1]
        
        if self.type is None:
            raise ValueError(
                "Model type (linear or logistic) must be specified."
            )
        
        nets = Network(
            size_w=self.n_agents,
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
            raise ValueError(f"Unknown network type: {self.net}")
    
    def _dpsgld_update(self, beta, X, y):
        for i in range(self.n_agents):
            for j in range(self.n_samples):
                if self.type == "linear":
                    grad = helpers.grad_BayesianLinearRegressionDPSGLD(
                        beta[i, :, j], X[i], y[i], self.gamma,
                        self.batch, self.lp, self.s, self.sigma, self.rng, self.n_agents
                    )
                elif self.type == "logistic":
                    grad = helpers.grad_BayesianLogisticRegressionDPSGLD(
                        beta[i, :, j], X[i], y[i], self.gamma,
                        self.batch, self.lp, self.s, self.rng, self.n_agents
                    )
                else:
                    raise NotImplementedError(
                        f"Model type {self.type} not implemented."
                    )
                temp = np.zeros(shape=(self.dim))
                for k in range(self.n_agents):
                    temp += self.W[i, k] * beta[k, :, j]
                noise = self.rng.standard_normal(self.dim)
                beta[i, :, j] = (
                    temp - self.eta * grad + np.sqrt(2 * self.eta) * noise
                )
        return beta
    
    def sample_parameters(
        self, iterations: int = None
    ):
        if iterations is not None:
            self.n_iters = iterations
        
        X = np.split(self.X, self.n_agents)
        y = np.split(self.y, self.n_agents)
        
        # Initialize the parameters from the prior
        betas = helpers.priors(
            dim=self.dim, s=self.s, lp=self.lp,
            N=self.n_samples * self.n_agents,
            rng=self.rng
        ).reshape(self.n_agents, self.dim, self.n_samples)
        
        history_all_agents = []
        beta_mean_all_agents = []
        for t in range(1):
            history = np.empty((self.n_agents, self.dim, self.n_samples))
            beta_mean = np.empty((self.dim, self.n_samples))
            for i in range(self.n_agents):
                history[i, :, :] = betas[i, :, :]
            for j in range(self.dim):
                beta_mean[j, :] = history[:, j, :].mean(axis=0)
            history_all_agents.append(history)
            beta_mean_all_agents.append(beta_mean)
        
        # Update the parameters (main loop)
        for t in tqdm(range(self.n_iters)):
            betas = self._dpsgld_update(betas, X, y)
            history = np.empty((self.n_agents, self.dim, self.n_samples))
            beta_mean = np.empty((self.dim, self.n_samples))
            for i in range(self.n_agents):
                history[i, :, :] = betas[i, :, :]
            for j in range(self.dim):
                beta_mean[j, :] = history[:, j, :].mean(axis=0)
            history_all_agents.append(history)
            beta_mean_all_agents.append(beta_mean)
            
        return np.array(history_all_agents), np.array(beta_mean_all_agents)
    
    
class DPSGLD1D:
    """
    Decentralized Proximal SGLD in one dimension
    
    Args: 
        n_agents (int): Number of agents in the decentralized network
        n_samples (int): Number of samples to generate
        n_iters (int): Number of iteration
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
    def __init__(self, n_agents, n_samples, eta, n_iters, dim=1,
                 mu=1.0, beta=0.5, b=1.0, sigma_grad=0.5,
                 gamma=1e-2, net='cn', R=1.0, seed=None):
        self.n_agents = n_agents
        self.n_samples = n_samples
        self.eta = eta
        self.n_iters = n_iters
        self.mu = mu
        self.beta = beta
        self.b = b
        self.sigma_grad = sigma_grad
        self.gamma = gamma
        self.net = net
        self.R = R
        self.dim = dim
        self.seed = seed

        # single RNG for the whole class
        self.rng = np.random.default_rng(seed)

        nets = Network(
            size_w=self.n_agents,
            seed=self.seed if self.seed is not None else 42
        )
        if self.net == 'fcn':
            self.W = nets.fully_connected_network()
        elif self.net == 'cn':
            self.W = nets.circular_network()
        elif self.net == 'sn':
            self.W = nets.star_network()
        elif self.net == 'fdn':
            self.W = nets.disconnected_network()
        else:
            raise ValueError(
                f'Network must "fcn", "cn", "sn", or "fdn", but got {self.net}'
            )

    def project_to_K(self, x):
        return np.clip(x, -self.R, self.R)

    def stochastic_gradient(self, x):
        grad_fi = self.mu * x + self.beta * x**3 - self.b
        noise = self.sigma_grad * self.rng.standard_normal()
        return grad_fi + noise

    def step(self, X):
        for n in range(self.n_samples):
            for i in range(self.n_agents):
                grad = self.stochastic_gradient(X[n, i])
                prox_grad = (
                    (X[n, i] - self.project_to_K(X[n, i])) / (self.gamma * self.n_agents)
                )
                Langevin_noise = (
                    np.sqrt(2.0 * self.eta) * self.rng.standard_normal()
                )
                temp = np.zeros(self.dim)
                for j in range(len(X[n])):
                    temp += self.W[i, j] * X[n, j]
                X[n, i] = (
                    temp - self.eta * (grad + prox_grad) + Langevin_noise
                )
        return X
    
    def sample(self):
        # Initialization
        X = np.random.normal(
            loc=0, scale=1, size=(self.n_samples, self.n_agents, self.dim)
        )
        
        history_all = []
        X_mean_all = []
        
        for t in range(1):
            history = np.empty((self.n_agents, self.dim, self.n_samples))
            X_mean = np.empty((self.dim, self.n_samples))
            for i in range(self.n_samples):
                history[:, :, i] = X[i, :, :]
            X_mean[0, :] = np.mean(history[:, 0, :], axis=0)
            history_all.append(history)
            X_mean_all.append(X_mean)
        # Update  
        for k in tqdm(range(self.n_iters)):
            X = self.step(X)
            history = np.empty((self.n_agents, self.dim, self.n_samples))
            X_mean = np.empty((self.dim, self.n_samples))
            
            for i in range(self.n_samples):
                history[:, :, i] = X[i, :, :]
            X_mean[0, :] = np.mean(history[:, 0, :], axis=0)
            history_all.append(history)
            X_mean_all.append(X_mean)
        
        return np.array(history_all), np.array(X_mean_all)