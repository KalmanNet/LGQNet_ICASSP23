import torch
import control
import numpy as np
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from model import f

from model import getJacobian

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")
   
   
def lqr_finite(horizon, f, G, QN, Qx, Qu, is_mismatch):
    '''
    Computes the LQR gains for the finite horizon problem with the
    given matrices.

    Parameters
    ----------
    horizon : int
    F : tensor of shape (m,m)
    G : tensor of shape (m,p)
    QN : tensor of shape (m,m)
    Qx : tensor of shape (m,m)
    Qu : tensor of shape (p,p)

    Returns
    -------
    L_list : list of tensors of shape (p,m) where L_list[0] is the lqr gain to 
    be used for input u[0]
    S_list : list of tensors of shape (m,m) where S_list[0] is the last element 
    of the backward recursion for S 
    '''
    # if is_mismatch:
    #     F = f(torch.eye(Qx.shape[1]), is_mismatch)
    # else:
    F = getJacobian(f(torch.tensor([[1],[1]],dtype=torch.float32)), 'ModAcc')
        
    FT = torch.transpose(F,1,0)
    GT = torch.transpose(G,1,0)
    S = QN
    L_list = []
    S_list = [S]
    for t in range(horizon):
        S1 = FT.matmul(S).matmul(F) + Qx
        S2 = FT.matmul(S).matmul(G) 
        S3 = GT.matmul(S).matmul(G) + Qu
        S3 = torch.pinverse(S3)

        S = S1 - S2.matmul(S3).matmul(torch.transpose(S2,1,0))
        S = (S + S.transpose(0,1)) / 2.0 # force symmetry
        S_list.append(S)
    S_list.reverse()

    for t in range(horizon):
        L1 = GT.matmul(S_list[t]).matmul(G) + Qu
        L1 = torch.pinverse(L1)
        L2 = GT.matmul(S_list[t]).matmul(F)
        L_list.append(L1.matmul(L2))
    
    return L_list , S_list


def lqr_infinite(f,G,Qx,Qu, is_mismatch):
    '''
    Description: 
        Discrete-Time Linear Quadratic Regulator (DLQR).
        Solve Algebraic Riccati Equation (ARE) and computes the control gain.
        Computes the control gain for both the finite/infinite time horizon.
    Output:
        L (2D array (or matrix)) State feedback gains
        S (2D array (or matrix)) Solution to Riccati equation
    '''
    F = getJacobian(f(torch.tensor([[1],[1]],dtype=torch.float32)), 'ModAcc')
        
    L, S, _ = control.dlqr(F.cpu(),G.cpu(),Qx.cpu(),Qu.cpu())
    return torch.FloatTensor(L).to(dev), torch.FloatTensor(S).to(dev)


def kalman_finite(T, F, H, Q, R, m2x_0):
    '''
    Computes the Kalman gains for the finite horizon problem with the
    given matrices.

    Parameters
    ----------
    horizon : int
    F : tensor of shape (m,m)
    G : tensor of shape (m,p)
    Q : tensor of shape (m,m)
        Process noise covariance
    R : tensor of shape (m,m)
        Observation noise covariance
    m2x_0 : tensor of shape (m,m)
        Covariance of initial state. If we know it, then set it to the zero matrix.

    Returns
    -------
    K_list : list of tensors of shape (n,m) where K_list[t] is the Kalman gain to 
    be used for y[t]
    P_list : list of tensors of shape (m,m) where P_list[t] is the error covariance 
    matrix of the estimate at time t
    '''
    # Computaten based on prediction and correction formulation
    FT = torch.transpose(F, 0, 1)
    HT = torch.transpose(H, 0, 1)
       
    # Initial Kalman gain
    K0 = torch.matmul(m2x_0, HT)
    K1 = H.matmul(m2x_0).matmul(HT) + R
    K2 = torch.inverse(K1)
    K = torch.matmul(K0, K2)

    # Initial error covariance matrix
    posterior = m2x_0 - K.matmul(K1).matmul(torch.transpose(K, 0, 1))

    P_list = [posterior]
    K_list = [K]
    for t in range(T):
        prior = F.matmul(posterior).matmul(FT) + Q
        S = H.matmul(prior).matmul(HT) + R
        Kt = prior.matmul(HT).matmul(torch.inverse(S))
        K_list.append(Kt)
        posterior = prior - Kt.matmul(S).matmul(torch.transpose(Kt,0,1))
        P_list.append(posterior)

    return K_list, P_list

################
### LQR Cost ###
################
def LQR_cost(SysModel, x, u):
        '''
        Computes the LQR cost for the given trajectories.

        Parameters
        ----------
        x : tensor
            State trajectory
        u : tensor
            Input trajectory
        xT : tensor
            Target state

        Returns
        -------
        cost : torch.float
        '''
        # TODO: add target trajectory
        
        T = max(u.shape)
        
        # Get design matrices from system model
        QT = SysModel.QT
        Qx = SysModel.Qx 
        Qu = SysModel.Qu 

        # Dimensions
        m = SysModel.m 
        n = SysModel.n
        
        # Scale the total cost by the time horizon
        scale = 1 / T

        dx = x #- xT.reshape(m, 1)

        terminal_cost = scale * torch.matmul(dx[:, -1], torch.matmul(QT, dx[:,-1]))

        stage_costs = 0
        for t in range(T):
            # x'Qx
            state_cost = torch.matmul(dx[:, t], torch.matmul(Qx, dx[:,t])) 
            # u'Ru
            control_cost = torch.matmul(u[:, t], torch.matmul(Qu, u[:,t]))
            stage_costs += scale * (state_cost + control_cost)

        cost = terminal_cost + stage_costs
        return cost



