import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

# # compute KL divergence loss from all Bayesian layers
# def compute_kl_divergence(model, verbose=False):
#     """
#     Compute the total KL divergence from all Bayesian layers in the model.
    
#     This function uses UQpy's GaussianKullbackLeiblerDivergence class to compute
#     KL divergence for all BayesianLinear layers in the model.
    
#     Parameters
#     ----------
#     model : nn.Module
#         The neural network model containing Bayesian layers
#     verbose : bool, optional
#         If True, print debug information about detected Bayesian layers
        
#     Returns
#     -------
#     kl_div : torch.Tensor
#         Total KL divergence from all Bayesian layers. Returns a zero tensor
#         if no Bayesian layers are found or if UQpy is not available.
#     """
#     # Try to import UQpy's classes
#     try:
#         import UQpy.scientific_machine_learning as sml
#         from UQpy.scientific_machine_learning.losses import GaussianKullbackLeiblerDivergence
#         BayesianLinear = sml.BayesianLinear
#     except ImportError:
#         if verbose:
#             print("[DEBUG] WARNING: UQpy not available, cannot compute KL divergence")
#         # Get device from model parameters if available
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = torch.device('cpu')
#         return torch.tensor(0.0, dtype=torch.float32, device=device)
    
#     # Count Bayesian layers for debugging
#     bayesian_layer_count = 0
#     for module in model.modules():
#         if isinstance(module, BayesianLinear):
#             bayesian_layer_count += 1
    
#     if verbose:
#         print(f"[DEBUG] Total Bayesian layers detected: {bayesian_layer_count}")
    
#     # Use UQpy's GaussianKullbackLeiblerDivergence class to compute KL divergence
#     # This follows the same approach as shown in UQpy documentation
#     try:
#         kl_divergence_fn = GaussianKullbackLeiblerDivergence(reduction="sum")
        
#         # Get device from model parameters
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = torch.device('cpu')
        
#         kl_divergence_fn.device = device
#         kl_div = kl_divergence_fn(model)
        
#         if verbose:
#             total_kl = kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div
#             print(f"[DEBUG] Total KL divergence computed: {total_kl:.6f}")
        
#         return kl_div
        
#     except Exception as e:
#         if verbose:
#             print(f"[DEBUG] Error computing KL divergence: {e}")
#             print("[DEBUG] Falling back to manual computation...")
        
#         # Fallback: manual computation using the same logic as UQpy
#         try:
#             import UQpy.scientific_machine_learning.functional as func
#             from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer
            
#             divergence = torch.tensor(0.0, dtype=torch.float32)
            
#             # Get device from model parameters
#             try:
#                 device = next(model.parameters()).device
#             except StopIteration:
#                 device = torch.device('cpu')
            
#             divergence = divergence.to(device)
            
#             for layer in model.modules():
#                 if not isinstance(layer, NormalBayesianLayer):
#                     continue
                
#                 if verbose and isinstance(layer, BayesianLinear):
#                     print(f"  [DEBUG] Computing KL for BayesianLinear layer")
                
#                 for name in layer.parameter_shapes:
#                     if layer.parameter_shapes[name] is None:
#                         continue
                    
#                     mu = getattr(layer, f"{name}_mu")
#                     rho = getattr(layer, f"{name}_rho")
                    
#                     layer_kl = func.gaussian_kullback_leibler_divergence(
#                         mu,
#                         torch.log1p(torch.exp(rho)),
#                         torch.tensor(layer.prior_mu, device=device),
#                         torch.tensor(layer.prior_sigma, device=device),
#                         reduction="sum",
#                     )
                    
#                     divergence += layer_kl
                    
#                     if verbose:
#                         kl_val_scalar = layer_kl.item() if isinstance(layer_kl, torch.Tensor) else layer_kl
#                         print(f"    [DEBUG]   Parameter '{name}' KL: {kl_val_scalar:.6f}")
            
#             if verbose:
#                 total_kl = divergence.item() if isinstance(divergence, torch.Tensor) else divergence
#                 print(f"[DEBUG] Total KL divergence (manual): {total_kl:.6f}")
            
#             return divergence
            
#         except Exception as e2:
#             if verbose:
#                 print(f"[DEBUG] Manual computation also failed: {e2}")
#             # Return zero tensor if all methods fail
#             try:
#                 device = next(model.parameters()).device
#             except StopIteration:
#                 device = torch.device('cpu')
#             return torch.tensor(0.0, dtype=torch.float32, device=device)
