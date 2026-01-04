import numpy as np


class WassersteinDistance1D:
    def __init__(self, R=1.0, 
                 history_all: np.ndarray=None,
                 X_mean_all: np.ndarray=None):
        self.R=R
        self.history_all=history_all
        self.X_mean_all=X_mean_all
        self.n_steps=history_all.shape[0]
        self.size_w=history_all.shape[1]
        
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
        
        w2_sq = np.mean((q - x_sorted)**2)
        
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
                d = self.Compute_W2distance(self.history_all[k,i,0,:], Q)
                w2dis[i].append(d)
                
        for k in range(self.n_steps):
            d = self.Compute_W2distance(self.X_mean_all[k,0,:], Q)
            w2dis[self.size_w].append(d)

        for i in range(len(w2dis)):
            w2dis[i]=np.array(w2dis[i])
        
        return w2dis
    
    def W2distSingleChain(self, chain, Q):
        chain = np.asarray(chain).reshape(-1)
        n = len(chain)
        
        w2 = np.empty(n, dtype=float)
        
        for k in range(n):
            w2[k] = self.Compute_W2distance(chain[:k+1], Q)
        
        return w2
        
            
        