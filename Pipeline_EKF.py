import torch
import torch.nn as nn
import random
from Extended_data import N_val
from Plot import Plot
import time
import os 
from lqr_utils import LQR_cost
import matplotlib.pyplot as plt
from support_functions import mean_and_std_linear_and_dB

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")


class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName #+ os.path.sep
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        
        # TODO: check if required
        self.LQR_ref_cost = 0
        self.LQG_ref_cost = 0 
        self.MSE_EKF_ref_dB = 0
        self.LQR_cost_true_system = 0
        self.LQG_cost_true_system = 0

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model.to(dev, non_blocking=True)

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay, alpha=1.0, beta=1.0, gamma=1.0):
        self.N_epochs       = n_Epochs      # Number of Training Epochs
        self.N_batch        = n_Batch       # Number of Samples in Batch
        self.learningRate   = learningRate  # Learning Rate
        self.weightDecay    = weightDecay   # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.mse_loss_fn    = nn.MSELoss(reduction='mean')

        # Training cost weights
        self.alpha          = alpha
        self.beta           = beta
        self.gamma          = gamma
        
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def NNTrain(self, training_noise, validation_noise, num_restarts=0):

        # Set training/validation size
        self.N_train = training_noise[0].shape[0]
        # self.N_val = validation_noise[0].shape[0] 
        self.N_val = self.N_batch 
        
        # Unpack noise 
        train_Q, train_R = training_noise
        val_Q, val_R = validation_noise

        # Make sure desired training trajectory length is feasible
        assert self.ssModel.T <= train_Q.shape[-1], "T should be less or equal the training size"
        
        # Set loss variables
        Total_loss_train_linear_batch = torch.empty([self.N_batch]) # linear scale
        self.Total_loss_train_linear_epoch = torch.empty([self.N_epochs]) # linear scale
        self.Total_loss_train_dB_epoch = torch.empty([self.N_epochs])

        Total_loss_val_batch = torch.empty([self.N_val])
        self.Total_loss_val_epoch = torch.empty([self.N_epochs])
        self.Total_loss_val_dB_epoch = torch.empty([self.N_epochs])

        LQR_val_linear_batch = torch.empty([self.N_val])
        self.LQR_val_linear_epoch = torch.empty([self.N_epochs])
        self.LQR_val_dB_epoch = torch.empty([self.N_epochs])

        MSE_val_batch = torch.empty([self.N_val])
        self.MSE_val_epoch = torch.empty([self.N_epochs])
        self.MSE_val_dB_epoch = torch.empty([self.N_epochs])

        MSE_val_position_batch = torch.empty([self.N_val])
        self.MSE_val_position_epoch = torch.empty([self.N_epochs])
        self.MSE_val_position_dB_epoch = torch.empty([self.N_epochs])
        
        # Set Loss parameters (init with very large numbers)
        self.Loss_val_dB_opt = 1000
        self.Loss_val_idx_opt = 0
        self.LQR_val_dB_opt = 1000
        self.LQR_val_idx_opt = 0
        self.MSE_val_dB_opt = 1000
        self.MSE_val_idx_opt = 0
        
        ##############
        ### Epochs ###
        ##############

        if num_restarts > 0:
            restart_every = int(self.N_epochs / (num_restarts+1))
            
        for ti in range(0, self.N_epochs):
            
            if num_restarts > 0:
                if ti % restart_every == 0:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_batch):
                
                # Select random noise sequence from training set
                idx = random.randint(0, self.N_train - 1)
                q_noise = train_Q[idx]
                r_noise = train_R[idx]

                # Tensors for state estimates and inputs
                x_hat = torch.empty(self.ssModel.m, self.ssModel.T + 1)
                x_hat[:,0] = self.ssModel.m1x_0
                x_true = torch.empty_like(x_hat)
                x_true[:,0] = self.ssModel.m1x_0
                
                # Pre allocate control input
                u = torch.zeros(self.ssModel.p, self.ssModel.T)
                
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)

                for t in range(1, self.ssModel.T+1):
                    
                    # Calculate LQR input
                    if self.model.is_control_enable:
                        dx = x_hat[:, t-1] #- XT[k]
                        if self.model.steady_state:
                            u[:, t-1] = - torch.matmul(self.ssModel.L, dx)
                        else:
                            u[:, t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                    
                    # Simulate state evolution + control
                    x_true[:, t] = self.ssModel.f(x_true[:,t-1],self.ssModel.is_mismatch) + self.ssModel.G.matmul(u[:, t-1]) + q_noise[:,t-1]
                    
                    # Simulate observation
                    yt = self.ssModel.h(x_true[:, t]) + r_noise[:,t-1]
                    
                    # Obtain state estimate from KalmanNet
                    x_hat[:,t] = self.model(yt, u[:,t-1])
                    

                # Compute loss for the trajectory
                Loss_lqr = LQR_cost(self.ssModel, x_true, u)
                Loss_mse = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:])

                # Weighted cose: alpha*MSE + beta*LQR
                Loss = self.alpha*Loss_mse + self.beta*Loss_lqr 
                Total_loss_train_linear_batch[j] = Loss.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + Loss


            # Average
            self.Total_loss_train_linear_epoch[ti] = torch.mean(Total_loss_train_linear_batch)
            self.Total_loss_train_dB_epoch[ti] = 10 * torch.log10(self.Total_loss_train_linear_epoch[ti])
            
            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_batch
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            with torch.no_grad():

                for j in range(0, self.N_val):

                    # Noise sequence to be used
                    q_noise = val_Q[j]
                    r_noise = val_R[j]

                    # Initialize simulation and KalmanNet
                    self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)

                    # Tensors for state estimates and inputs
                    x_hat = torch.empty(self.ssModel.m, self.ssModel.T + 1)
                    x_hat[:,0] = self.ssModel.m1x_0
                    x_true = torch.empty_like(x_hat)
                    x_true[:,0] = self.ssModel.m1x_0
                    # Pre allocate control input
                    u = torch.zeros(self.ssModel.p, self.ssModel.T)

                    # Simulate trajectory
                    for t in range(1, self.ssModel.T + 1):
                        
                        # Calculate LQR input
                        if self.model.is_control_enable:
                            dx = x_hat[:, t-1] #- XT[k]
                            if self.model.steady_state:
                                u[:, t-1] = - torch.matmul(self.ssModel.L, dx)
                            else:
                                u[:, t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                        
                        # Simulate state evolution + control
                        x_true[:, t] = self.ssModel.f(x_true[:,t-1], self.ssModel.is_mismatch) + self.ssModel.G.matmul(u[:, t-1]) + q_noise[:,t-1]
                        
                        # Simulate observation
                        yt = self.ssModel.h(x_true[:, t]) + r_noise[:,t-1]
                        
                        # Obtain state estimate from KalmanNet
                        x_hat[:,t] = self.model(yt, u[:,t-1])

                    # Compute LQR Loss
                    LQR_val_linear_batch[j] = LQR_cost(self.ssModel, x_true, u)
                    # MSE of state estimation
                    MSE_val_batch[j] = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:]).item()
                    MSE_val_position_batch[j] = self.mse_loss_fn(x_hat[0,1:], x_true[0,1:]).item()

                    # Total loss: MSE + LQR
                    Total_loss_val_batch[j] = self.alpha*MSE_val_batch[j] + self.beta*LQR_val_linear_batch[j]


                # Average losses
                self.LQR_val_linear_epoch[ti] = torch.mean(LQR_val_linear_batch)
                self.LQR_val_dB_epoch[ti] = 10 * torch.log10(self.LQR_val_linear_epoch[ti])

                self.MSE_val_epoch[ti] = torch.mean(MSE_val_batch)
                self.MSE_val_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_epoch[ti])
                self.MSE_val_position_epoch[ti] = torch.mean(MSE_val_position_batch)
                self.MSE_val_position_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_position_epoch[ti])

                self.Total_loss_val_epoch[ti] = torch.mean(Total_loss_val_batch)
                self.Total_loss_val_dB_epoch[ti] = 10 * torch.log10(self.Total_loss_val_epoch[ti])

                # Save model in case of improvement
                if (self.Total_loss_val_dB_epoch[ti] < self.Loss_val_dB_opt):
                    self.Loss_val_dB_opt = self.Total_loss_val_dB_epoch[ti]
                    self.Loss_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName)

                # Save best LQR model
                if (self.LQR_val_dB_epoch[ti] < self.LQR_val_dB_opt):
                    self.LQR_val_dB_opt = self.LQR_val_dB_epoch[ti]
                    self.LQR_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName[:-3] + '_best_LQR.pt')

                # Save best MSE model
                if (self.MSE_val_dB_epoch[ti] < self.MSE_val_dB_opt):
                    self.MSE_val_dB_opt = self.MSE_val_dB_epoch[ti]
                    self.MSE_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName[:-3] + '_best_MSE.pt')


            ########################
            ### Training Summary ###
            ########################
            
            if (ti > 0):
                d_val = self.Total_loss_val_dB_epoch[ti] - self.Total_loss_val_dB_epoch[ti - 1]
                d_mse = self.MSE_val_dB_epoch[ti] - self.MSE_val_dB_epoch[ti - 1] 
                d_lqr = self.LQR_val_dB_epoch[ti] - self.LQR_val_dB_epoch[ti - 1]
                info = f"{ti} LQG train: {self.Total_loss_train_dB_epoch[ti]: .5f} [dB], " \
                        f"LQG val: {self.Total_loss_val_dB_epoch[ti]: .5f} [dB], " \
                        f"LQR val: {self.LQR_val_dB_epoch[ti]: .5f} [dB], " \
                        f"MSE val: {self.MSE_val_dB_epoch[ti]: .5f} [dB]" \
                        f"diff LQG val: {d_val: .5f} [dB], diff LQR val: {d_lqr: .5f} [dB] , diff MSE val: {d_mse: .5f} [dB] " \
                        f"best idx: {self.Loss_val_idx_opt}, Best cost: {self.Loss_val_dB_opt: .5f} [dB] " \
                        f"best idx MSE: {self.MSE_val_idx_opt}, best MSE: {self.MSE_val_dB_opt: .5f} [dB]" \
                        f"best idx LQR: {self.LQR_val_idx_opt}, best LQR: {self.LQR_val_dB_opt: .5f} [dB]"
                print(info)
            else:
                print(f"{ti} LQG train : {self.Total_loss_train_dB_epoch[ti]: .5f} [dB], LQG val : {self.Total_loss_val_dB_epoch[ti]: .5f} [dB]")

            # If loss is nan stop
            if self.Total_loss_train_dB_epoch[ti].isnan():
                break
        

    def NNTest(self, test_noise):
        
        # Unpack noise 
        test_Q, test_R = test_noise
        
        self.N_test = test_Q.shape[0]

        self.LQR_test_linear_arr = torch.empty([self.N_test])
        self.MSE_test_arr = torch.empty([self.N_test])
        self.MSE_test_position_arr = torch.empty([self.N_test])
             
        self.model = torch.load(self.modelFileName, map_location=dev)
        
        self.model.eval()
        
        with torch.no_grad():
        
            start = time.time()
            for j in range(0, self.N_test):
                q_noise = test_Q[j]
                r_noise = test_R[j]

                # Initialize X0
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T_test)

                # Tensors for state estimates and inputs
                x_hat = torch.empty(self.ssModel.m, self.ssModel.T_test + 1)
                x_hat[:,0] = self.ssModel.m1x_0
                x_true = torch.empty_like(x_hat)
                x_true[:,0] = self.ssModel.m1x_0
                
                # Pre allocate control input
                u = torch.zeros(self.ssModel.p, self.ssModel.T_test)
                
                for t in range(1, self.ssModel.T_test+1):
                    
                    # Calculate LQR input
                    if self.model.is_control_enable:
                        dx = x_hat[:, t-1] #- XT[k]
                        if self.model.steady_state:
                            u[:, t-1] = - torch.matmul(self.ssModel.L, dx)
                        else:
                            u[:, t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                    
                    # Simulate state evolution + control
                    x_true[:, t] = self.ssModel.f(x_true[:,t-1], self.ssModel.is_mismatch) + self.ssModel.G.matmul(u[:, t-1]) + q_noise[:,t-1]
                    
                    # Simulate observation
                    yt = self.ssModel.h(x_true[:, t]) + r_noise[:,t-1]
                    
                    # Obtain state estimate from KalmanNet
                    x_hat[:,t] = self.model(yt, u[:,t-1])
                            
                
                # Compute loss for the trajectory
                self.LQR_test_linear_arr[j] = LQR_cost(self.ssModel, x_true, u)
                
                # MSE of state estimate
                self.MSE_test_arr[j] = self.mse_loss_fn(x_hat[:,0:self.ssModel.T_test], x_true[:,0:self.ssModel.T_test])
                self.MSE_test_position_arr[j] = self.mse_loss_fn(x_hat[0,0:self.ssModel.T_test], x_true[0,0:self.ssModel.T_test])
        
            end = time.time()
            t = end - start

            # Average and standard deviation
            self.LQR_test_linear_avg, self.LQR_test_dB_avg, self.LQR_test_std, self.LQR_test_dB_std = mean_and_std_linear_and_dB(self.LQR_test_linear_arr)
            self.MSE_test_avg, self.MSE_test_dB_avg, self.MSE_test_std, self.MSE_test_dB_std = mean_and_std_linear_and_dB(self.MSE_test_arr)
            self.MSE_test_position_avg, self.MSE_test_position_dB_avg, self.MSE_test_position_std, self.MSE_test_position_dB_std = mean_and_std_linear_and_dB(self.MSE_test_position_arr)

        print(f"{self.modelName} - LQR Test: {self.LQR_test_dB_avg} [dB], STD: {self.LQR_test_dB_std} [dB]")
        print(f"{self.modelName} - MSE Test: {self.MSE_test_dB_avg} [dB], STD: {self.MSE_test_dB_std} [dB]")
        print(f"{self.modelName} - Position MSE Test: {self.MSE_test_position_dB_avg} [dB], STD: {self.MSE_test_position_std} [dB]")
        print("Inference Time:", t)

        LQR_loss_summary = (self.LQR_test_linear_arr, self.LQR_test_linear_avg, self.LQR_test_dB_avg)
        MSE_loss_total_summary = (self.MSE_test_arr, self.MSE_test_avg, self.MSE_test_dB_avg)
        MSE_loss_position_summary = (self.MSE_test_position_arr, self.MSE_test_position_avg, self.MSE_test_position_dB_avg)

        return LQR_loss_summary, MSE_loss_total_summary, MSE_loss_position_summary

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)