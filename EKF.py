"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from filing_paths import path_model
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, path_model)
print(sys.path)
from model import getJacobian

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, mode='full', steady_state=False):
        
        ####################
        ### Motion Model ###
        ####################
        self.f = SystemModel.f
        self.m = SystemModel.m

        self.G = SystemModel.G
        self.p = SystemModel.p
        
        # Model mismatch flag
        self.is_mismatch = SystemModel.is_mismatch
        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        #########################
        ### Observation Model ###
        #########################
        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T_test,self.m,self.n))

        # Full knowledge about the model or partial? (Should be made more elegant)
        if(mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif(mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'
            
        # Set control gain
        if steady_state:
            self.L = SystemModel.L_infinite # Corresponds to infinite dlqr
        else:
            self.L = SystemModel.L          # Corresponds to finite dlqr
            
   
    # Predict
    def Predict(self, u):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior)) + self.G.matmul(u)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.fString), getJacobian(self.m1x_prior, self.hString))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior))
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y, u):
        self.Predict(u)
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F,0,1)
        self.H = H
        self.H_T = torch.transpose(H,0,1)
        #print(self.H,self.F,'\n')
        
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, T, q_noise, r_noise, steady_state=False, is_control_enable=True, EKF_enable=True):
        
        # Pre allocate an array for estimated state 
        self.x_hat = torch.empty(size=[self.m, T+1])
        self.x_hat[:,0] = torch.squeeze(self.m1x_0)
        # Pre allocate an array for actual state and variance
        self.x = torch.empty(size=[self.m, T+1])
        self.x[:,0] = torch.squeeze(self.m1x_0)
        # Pre allocate estimated state's variance
        self.sigma = torch.empty(size=[self.m, self.m, T+1])
        self.sigma[:,:,0] = torch.squeeze(self.m2x_0)
        
        # Pre allocate KG array
        self.KG_array = torch.zeros((T,self.m,self.n))
        self.i = 0 # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T+1])
        self.y[:,0] = self.h(self.m1x_0)
        
        # Get control gain
        L = self.L
        # Pre allocate control input
        self.u = torch.zeros(self.p, T)
        
        # Debug params
        condVec = torch.zeros(1,T)
        
        
        for t in range(1, T+1):
            
            if is_control_enable:
                # LQR input
                dx = self.x_hat[:, t-1] #- XT[k]
                if steady_state:
                    self.u[:, t-1] = - torch.matmul(L, dx)
                else:
                    self.u[:, t-1] = - torch.matmul(L[t-1], dx)
                    
            # State Model
            self.x[:,t] = self.f(self.x[:, t-1], self.is_mismatch) + self.G.matmul(self.u[:, t-1]) + q_noise[:, t-1] 
            # Observation model
            yt = self.h(self.x[:,t], self.is_mismatch) + r_noise[:, t-1]
            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)
            
            if EKF_enable:
                # Run EKF
                self.x_hat[:, t], self.sigma[:, :, t] = self.Update(yt, self.u[:, t-1])
                # Compute condition number of estimated 2nd moment state
                condVec[:,t-1] = torch.linalg.cond(self.sigma[:,:,t]) 
            else:
                # EKF is disabled
                self.x_hat[:, t] = self.x[:,t]
                
        
        # Omit the first time step
        # self.x_hat = self.x_hat[:, 1:]
        # self.x = self.x[:, 1:]
        self.x_hat = self.x_hat[:, 0:T]
        self.x = self.x[:, 0:T]
        self.sigma = self.sigma[:,:, 1:]
        self.y = self.y[:, 1:]