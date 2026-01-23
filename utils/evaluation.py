import numpy as np
from scipy.linalg import sqrtm
import math


class WassersteinDistance1D:
    def __init__(
        self, R=1.0, history_all: np.ndarray = None,
        X_mean_all: np.ndarray = None
    ):
        self.R = R
        self.history_all = history_all
        self.X_mean_all = X_mean_all
        self.n_steps = history_all.shape[0]
        self.size_w = history_all.shape[1]

        if (history_all is None) or (X_mean_all is None):
            raise ValueError("history_all or X_mean_all is required, got None")

    def Compute_W2distance(self, samples, Q):
        x = np.asarray(samples)

        x_sorted = np.sort(x)
        n = x_sorted.size

        if n == 0:
            raise ValueError("Empty sample array")

        u = (np.arange(n) + 0.5) / n
        q = Q(u)

        w2_sq = np.mean((q - x_sorted) ** 2)

        return float(np.sqrt(w2_sq))

    def W2dist(self, Q):
        w2dis = []
        for i in range(self.size_w):
            temp = []
            w2dis.append(temp)
        temp = []
        w2dis.append(temp)

        for i in range(self.size_w):
            for k in range(self.n_steps):
                d = self.Compute_W2distance(self.history_all[k, i, 0, :], Q)
                w2dis[i].append(d)

        for k in range(self.n_steps):
            d = self.Compute_W2distance(self.X_mean_all[k, 0, :], Q)
            w2dis[self.size_w].append(d)

        for i in range(len(w2dis)):
            w2dis[i] = np.array(w2dis[i])

        return w2dis

    def W2distSingleChain(self, chain, Q):
        """
        chain: array of shape (T, N)  (T = n_steps)
        returns: W2 distance per iteration, length T
        """
        chain = np.asarray(chain)

        if chain.ndim != 2:
            raise ValueError(
                f"Expected chain with shape (T,N), got shape {chain.shape}"
            )

        T = chain.shape[0]
        w2dis = []
        for k in range(T):
            d = self.Compute_W2distance(chain[k, :], Q)
            w2dis.append(d)

        return np.array(w2dis)


class Wasserstein2Distance:
    """Class for: Wasserstein 2 distance in Bayesian Linear Regression

    Args:
        size_w (int): the size of the network
        T (int): the number of iterations
        avg_post (list): the mean of the posterior distribution
        cov_post (list): the covariance of the posterior distribution
        history_all (list): contains the approximation from all the nodes
        beta_mean_all (list): contains the mean of the approximation
        from all the nodes
    """

    def __init__(
        self, size_w, T, avg_post, cov_post, history_all, beta_mean_all
    ):
        self.size_w = size_w
        self.T = T
        self.avg_post = avg_post
        self.cov_post = cov_post
        self.history_all = history_all
        self.beta_mean_all = beta_mean_all

    def W2_dist(self):
        """Class for: Wasserstein 2 distance in Bayesian Linear Regression

        Args:
            size_w (int): the size of the network
            T (int): the number of iterations
            avg_post (list): the mean of the posterior distribution
            cov_post (list): the covariance of the posterior distribution
            history_all (list): contains the approximation from all the nodes
            beta_mean_all (list): contains the mean of the approximation
            from all the nodes
        Returns:
        w2dis (list): contains the W2 distance of each agent and
        the mean of the approximation from all the agents
        """
        w2dis = []
        for i in range(self.size_w):
            temp = []
            w2dis.append(temp)
        temp = []
        w2dis.append(temp)
        """
        W2 distance of each agent
        """
        for i in range(self.size_w):
            for t in range(self.T + 1):
                d = 0
                avg_temp = []
                avg_temp.append(np.mean(self.history_all[t][i][0]))
                avg_temp.append(np.mean(self.history_all[t][i][1]))
                avg_temp = np.array(avg_temp)
                cov_temp = np.cov(self.history_all[t][i])
                d = np.linalg.norm(self.avg_post - avg_temp) * np.linalg.norm(
                    self.avg_post - avg_temp
                )
                d = d + np.trace(
                    self.cov_post
                    + cov_temp
                    - 2
                    * sqrtm(
                        np.dot(
                            np.dot(sqrtm(cov_temp), self.cov_post),
                            sqrtm(cov_temp),
                        )
                    )
                )
                w2dis[i].append(np.array(math.sqrt(abs(d))))
        """
        W2 distance of the mean of agents
        """
        for t in range(self.T + 1):
            d = 0
            avg_temp = []
            avg_temp.append(np.mean(self.beta_mean_all[t][0]))
            avg_temp.append(np.mean(self.beta_mean_all[t][1]))
            avg_temp = np.array(avg_temp)
            cov_temp = np.cov(self.beta_mean_all[t])
            d = np.linalg.norm(self.avg_post - avg_temp) * np.linalg.norm(
                self.avg_post - avg_temp
            )
            d = d + np.trace(
                self.cov_post
                + cov_temp
                - 2
                * sqrtm(
                    np.dot(
                        np.dot(sqrtm(cov_temp), self.cov_post), sqrtm(cov_temp)
                    )
                )
            )
            w2dis[self.size_w].append(np.array(math.sqrt(abs(d))))

        for i in range(len(w2dis)):
            w2dis[i] = np.array(w2dis[i])

        return w2dis


class AccuracyEvaluator:
    def __init__(self, chain):
        """  
        Accepts chain of shape (T, w, dim, N) or (T, dim, N)
        where 
            T: Number of iterations 
            w: Size of the communication matrix (inherited from
               the chain)
            dim: Dimension of the sampling problem
            N: Number of samples in each iteration
        """
        if chain.ndim == 4:
            T, w, dim, N = chain.shape
            # Generate and store the random indices (0 to w-1)
            # size=T means we pick one random w for every timestep t
            self.selected_agent = np.random.randint(0, w)
            # Extract weights using advanced indexing
            self.weights = chain[:, self.selected_agent, :, :]
        else:
            self.weights = chain
    
    def get_selected_agent(self):
        return self.selected_agent
    
    def compute_accuracy(self, X, y):
        """Vectorize accuracy calculation"""
        # weights shape (T, dim, N) --> (T, N, dim)
        w_tensor = np.transpose(self.weights, (0, 2, 1))
        
        # (T, N, dim) @ (dim, M) -> (T, N, M)
        logits = np.matmul(w_tensor, X.T)
        
        # vectorize the predictions
        preds = (logits >= 0).astype(int)
        is_correct = (preds == y[None, None, :])
        acc_per_run = np.mean(is_correct, axis=2)
        return np.mean(acc_per_run, axis=1), np.std(acc_per_run, axis=1)