# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license. Extracted conditional inference prior.
import torch
from scipy.spatial.transform import Rotation
from flexgen._vendor.gafl.data import so3_utils

class FrameInterpolant:
    def __init__(self,cfg):
        self._cfg=cfg
        if cfg.prior!='conditional' or cfg.batch_ot!='none' or cfg.rots.distribution!='uniform':
            raise ValueError('Only the paper conditional prior is supported')
    def set_device(self,device):
        self._device=device
    def sample_from_prior(self,num_batch,num_res,trans_equilibrium,rotmats_equilibrium):
        x=torch.randn(num_batch,num_res,3,device=self._device)
        x=x-torch.mean(x,dim=-2,keepdim=True)
        x=x*10
        x*=self._cfg.noise_scale
        if self._cfg.noise_res_scaling_power>0:
            x*=(num_res/128.)**self._cfg.noise_res_scaling_power
        r=torch.tensor(Rotation.random(num_batch*num_res).as_matrix(),device=self._device,dtype=torch.float32).reshape(num_batch,num_res,3,3)
        t=self._cfg.prior_conditional.gamma_trans*torch.ones(num_batch,1,device=self._device,dtype=torch.float32)
        x=(1-t[...,None])*x+t[...,None]*trans_equilibrium
        t=self._cfg.prior_conditional.gamma_rots*torch.ones(num_batch,1,device=self._device,dtype=torch.float32)
        r=so3_utils.geodesic_t(t[...,None],rotmats_equilibrium,r)
        return x,r
