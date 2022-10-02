import torch.nn as nn
import torch
import time
from EKF import ExtendedKalmanFilter
from lqr_utils import LQR_cost
import matplotlib.pyplot as plt

def EKFTest(SysModel, noise, modelKnowledge = 'full', allStates=True, steady_state=False, is_control_enable=True, EKF_enable=True):

    # Get Process/Observation Noise
    Q_noise, R_noise = noise
    
    # Get Trajectory length
    N_T = SysModel.T_test

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    
    # LQG Linear scale
    LQG_EKF_arr = torch.empty(N_T)
    
    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge, steady_state=steady_state)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_x_hat = torch.empty([N_T, SysModel.m, SysModel.T_test])
    EKF_x_true = torch.empty([N_T, SysModel.m, SysModel.T_test])
    
    start = time.time()
    for j in range(0, N_T):
        
        q_noise = Q_noise[j]
        r_noise = R_noise[j]
        
        EKF.GenerateSequence(EKF.T_test, q_noise, r_noise, is_control_enable=is_control_enable, EKF_enable=EKF_enable)
        
        # if(allStates):
        MSE_EKF_linear_arr[j] = loss_fn(EKF.x_hat, EKF.x).item()
        
        KG_array = torch.add(EKF.KG_array, KG_array) 
        EKF_x_hat[j,:,:] = EKF.x_hat
        EKF_x_true[j,:,:] = EKF.x
        # Compute LQR cost per example
        LQG_EKF_arr[j] = LQR_cost(SysModel, EKF.x, EKF.u)
        
    trKG = torch.empty(len(KG_array))    
    for i in range(len(KG_array)):
        trKG[i] = EKF.KG_array[i].trace()/2
        
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)
    
    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    

    # Calculate LQG cost given estimated trajectory + control
    # TODO: add target state - XT
    LQG_avg     = torch.mean(LQG_EKF_arr)
    LQG_cost_dB = 10*torch.log10(LQG_avg)
    
    if EKF_enable:
        print("LQG cost:", LQG_cost_dB, "[dB]")
    else:
        print("LQR cost:", LQG_cost_dB, "[dB]")
        
    # Print Run Time
    print("Inference Time:", t)
    
    return [MSE_EKF_dB_avg, MSE_EKF_dB_std, LQG_cost_dB, EKF_x_hat, EKF_x_true]



