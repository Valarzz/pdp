import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from numba import jit

@jit(nopython=True)
def compute_softdtw_cpu(D, gamma, bandwidth):
    """
    CPU implementation of soft-DTW forward pass
    """
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    
    for b in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue
                
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    
    return R

@jit(nopython=True)
def compute_softdtw_backward_cpu(D_, R, gamma, bandwidth):
    """
    CPU implementation of soft-DTW backward pass
    """
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                
                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue
                
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    
    return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
    """
    Differentiable soft-DTW implementation
    """
    
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.tensor([gamma]).to(dev).type(dtype)
        bandwidth = torch.tensor([bandwidth]).to(dev).type(dtype)
        
        D_cpu = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        
        R = torch.tensor(compute_softdtw_cpu(D_cpu, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]
    
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        
        D_cpu = D.detach().cpu().numpy()
        R_cpu = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        
        E = torch.tensor(compute_softdtw_backward_cpu(D_cpu, R_cpu, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

class SoftDTW(nn.Module):
    """
    Soft DTW module for differentiable DTW computation
    """
    
    def __init__(self, gamma=1.0, bandwidth=None, normalize=False):
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.normalize = normalize
    
    def _euclidean_dist_func(self, x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)
    
    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed soft-DTW distances
        """
        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self._euclidean_dist_func(x, y)
            out = _SoftDTW.apply(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 0.5 * (out_xx + out_yy)
        else:
            D_xy = self._euclidean_dist_func(X, Y)
            return _SoftDTW.apply(D_xy, self.gamma, self.bandwidth)

def soft_dtw_distance(x, y, gamma=1.0, bandwidth=None):
    """
    Convenience function to compute soft-DTW distance between two sequences
    """
    # Ensure inputs have batch dimension
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    
    dtw = SoftDTW(gamma=gamma, bandwidth=bandwidth)
    return dtw(x, y).squeeze(0) if x.size(0) == 1 else dtw(x, y) 