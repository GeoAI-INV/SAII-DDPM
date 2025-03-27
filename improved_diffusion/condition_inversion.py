'by Hongling Chen, Xian Jiaotong university'

import torch
from abc import ABC
import pylops
import numpy as np
from improved_diffusion.Forward_oper import *


def fanyan_pylop(record, wav, x0, regpara):
    x0 = x0.numpy()
    x = np.ones_like(x0)
    for a in range(x0.shape[0]):
        x[a,0,:,:]= pylops.avo.poststack.PoststackInversion(
            record, wav, m0=x0[a,0,:,:], explicit=True, simultaneous=False, epsI=regpara)[0]
    x = torch.tensor(x)
    return x


def fanyan_pylop_tv(record, wav, x0, regpara):
    [nz, nx] = record.shape

    Dop2D = [pylops.FirstDerivative(nz * nx, dims=(nz, nx), dir=0, edge=False, kind='backward'),
             pylops.FirstDerivative(nz * nx, dims=(nz, nx), dir=1, edge=False, kind='backward')]
    PPop2D = pylops.avo.poststack.PoststackLinearModelling((wav) / 2, nt0=nz,
                                                           spatdims=nx) # 注意这个位置，不应该除以2！！！
    x0 = x0.numpy()
    x = np.ones_like(x0)
    for  a in range(x0.shape[0]):
       x[a,0,:,:] = \
            pylops.optimization.sparsity.SplitBregman(PPop2D, Dop2D, record.ravel(),
                                                  niter_outer=10, niter_inner=3,
                                                  mu=1, epsRL1s=[0.05, 0.05],
                                                  tol=1e-3, tau=1, x0=x0[a,0,:,:].flatten(), show=False,
                                                  **dict(iter_lim=5, damp=1e-3))[0].reshape(x0[0,0,:,:].shape)
    x = torch.tensor(x)
    return x



class ConditioningMethod(ABC):
    def __init__(self, operator):
        self.operator = operator
    def grad_and_value(self, imp0, measurement):  # 重点位置
        difference = measurement - self.operator.forward(imp0)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=imp0)[0]
        return norm_grad, norm


class PosteriorSample_chl(ConditioningMethod):
    def __init__(self, operator):
        super().__init__(operator)
    def conditioning(self, imp0, measurement):
        imp0 = imp0.requires_grad_(True)
        norm_grad, norm = self.grad_and_value(imp0=imp0, measurement=measurement)
        return norm_grad

    def compute_dps(self, y:torch.Tensor, x_0:torch.Tensor, x_t:torch.Tensor):
        difference = (y - self.operator(x_0))
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
        return norm_grad

    def compute_dps_new(self, y:torch.Tensor, x_0:torch.Tensor, x_t:torch.Tensor, x_l:torch.Tensor, scale:float):
        difference = (y - self.operator(x_0))
        difference_l = x_0-x_l
        norm = torch.linalg.norm(difference)
        norm_l = scale*torch.linalg.norm(difference_l)
        norm_new = norm + norm_l
        norm_grad = torch.autograd.grad(outputs=norm_new, inputs=x_t)[0]
        return norm_grad




