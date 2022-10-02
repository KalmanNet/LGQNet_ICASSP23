
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from lqr_utils import lqr_finite, kalman_finite, lqr_infinite

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

class SystemModel:

    def __init__(self, f, G, Q, h, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None, is_mismatch=False):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.G = G
        self.p = self.G.size()[1]
        
        self.Q = Q
        self.m = self.Q.size()[0]

        # Model mismatch flag
        self.is_mismatch = is_mismatch
        
        #########################
        ### Observation Model ###
        #########################
        self.h = h

        self.R = R
        self.n = self.R.size()[0]

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.eye(self.m)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S



    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = torch.squeeze(m1x_0).to(dev) #m1x_0
        self.x_prev = torch.squeeze(m1x_0).to(dev) #m1x_0
        self.m2x_0 = torch.squeeze(m2x_0).to(dev) #m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, T, q_noise, r_noise, steady_state=False, is_control_enable=True):
        
        # Get control gain
        if steady_state:
            L = self.L_infinite # Corresponds to infinite dlqr
        else:
            L = self.L          # Corresponds to finite dlqr
            
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T+1])
        self.x[:,0] = self.m1x_0
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T+1])
        self.y[:,0] = self.h(self.x[:,0])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        # Pre allocate control
        u = torch.zeros(self.p, T)
        
        # xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):
            
            ########################
            ##### Control Input ####
            ########################
            if is_control_enable:
                # LQR input
                dx = self.x[:, t-1] # - XT[k]
                if steady_state:
                    u[:, t-1] = - torch.matmul(L, dx)
                else:
                    u[:, t-1] = - torch.matmul(L[t-1], dx)
                    
            ########################
            #### State Evolution ###
            ########################
            xt = self.f(self.x_prev, self.is_mismatch) + self.G.matmul(u[:, t-1]) + q_noise[:,t-1]
            
            ################
            ### Emission ###
            ################
            yt = self.h(xt) + r_noise[:,t-1]
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, Q_noise, R_noise, randomInit=False, seqInit=False, T_test=0, steady_state=False, is_control_enable=True):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        # Generate Sequence
        for i in range(0, size):
            # Noise sequence    
            q_noise = Q_noise[i]
            r_noise = R_noise[i]
            
            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(T, q_noise, r_noise, steady_state=steady_state, is_control_enable=is_control_enable)

            
            # Training sequence input
            self.Input[i, :, :] = self.y[:,1:]
            # Training sequence output
            self.Target[i, :, :] = self.x[:,1:]


    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = np.transpose(Aq) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = np.transpose(Ar) * Ar

        return [Q_gen, R_gen]

    ######################
    ######## LQR #########
    ######################

    def InitCostMatrices(self, QN, Qx, Qu):
        self.QT = QN
        self.Qx = Qx
        self.Qu = Qu

    def ComputeLQRgains(self):
        # self.L, self.S = lqr_finite(self.T, self.f, self.G, self.QT, self.Qx, self.Qu, self.is_mismatch)
        self.L, self.S = lqr_finite(self.T_test, self.f, self.G, self.QT, self.Qx, self.Qu, self.is_mismatch)
        self.L_infinite, self.S_infinite = lqr_infinite(self.f, self.G, self.Qx, self.Qu, self.is_mismatch)
        # if isinstance(self.system, LinearSystem):
        #     self.L_true, self.S_true = lqr_finite(
        #         self.T, self.system.F, self.system.G, self.QT, self.Qx, self.Qu)
        #     self.L_infinite_true, self.S_infinite_true = lqr_infinite(
        #         self.system.F, self.system.G, self.Qx, self.Qu)
        # else:
        #     self.L_true, self.S_true = self.L, self.S
        #     self.L_infinite_true, self.S_infinite_true = self.L_infinite, self.S_infinite
        
        
        
    def GenNoiseSequence(self, T, N_samples):
        
        # Allocate storage for data
        seq_Q = torch.empty(N_samples, self.m, T)
        seq_R = torch.empty(N_samples, self.n, T)
        
        # Distributions to sample from 
        distrib_Q = MultivariateNormal(loc=torch.zeros([self.m]), covariance_matrix=self.Q)
        distrib_R = MultivariateNormal(loc=torch.zeros([self.n]), covariance_matrix=self.R)

        for k in range(N_samples):
            for t in range(T):
                q_sample = distrib_Q.rsample()
                r_sample = distrib_R.rsample()
                seq_Q[k,:,t] = q_sample
                seq_R[k,:,t] = r_sample

        noise = (seq_Q, seq_R)
        
        return noise 