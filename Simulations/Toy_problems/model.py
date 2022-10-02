import autograd.numpy as np
from autograd import grad, jacobian
import torch
from torch import autograd

from parameters import alpha_mot, beta_mot, phi_mot, a_mot, alpha_obs, beta_obs, a_obs
from parameters import alpha_mot_mod, beta_mot_mod, phi_mot_mod, a_mot_mod, beta_obs_mod, a_obs_mod
from math import log10, cos, sin, pi

def f(x, is_mismatch = False):
    
    # f = alpha_mot * torch.sin(beta_mot * x + phi_mot) + a_mot
    f = torch.tensor([[1, 1], [0,  1]], dtype=torch.float32).matmul(x)
    if is_mismatch:
        a_deg = 20
        a = a_deg / 180 * pi
        Rot = torch.tensor([[cos(a), -sin(a)], [sin(a),  cos(a)]])
        f = torch.matmul(Rot, f)
        
        
    return f

def h(x):
    # h = alpha_obs * (beta_obs*x + a_obs)**2
    # x2 = 0.0001 * (x.copy())**2
    # h = 1 / ( 1 + torch.exp(-x))
    h = x
    return h

def fInacc(x):
    return alpha_mot_mod * torch.sin(beta_mot_mod * x + phi_mot_mod) + a_mot_mod

def hInacc(x):
    return alpha_obs_mod * (beta_obs_mod*x + a_obs_mod)**2

def getJacobian(x, a):
    
    try:
        if(x.size()[1] == 1):
            y = torch.reshape((x.T),[x.size()[0]])
    except:
        y = torch.reshape((x.T),[x.size()[0]])

    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = hInacc
    elif(a == 'ModInacc'):
        g = fInacc

    return autograd.functional.jacobian(g, y)
