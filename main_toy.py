import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from KalmanNet_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split, NoiseGen, NoiseLoader_GPU
# from Extended_data import N_E, N_CV, N_T
from Extended_data import N_train, N_val, N_test
from Pipeline_EKF import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN
from datetime import datetime
# from Plot import Plot
from Plot_copy import Plot

from EKF_test import EKFTest
# from UKF_test import UKFTest
# from PF_test import PFTest

from filing_paths import path_model
import os
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n, Qx, Qu, QT, G
from model import f, h, fInacc

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

   

print("Pipeline Start")
num = 0
# torch.random.seed(num)
print('set seed to be {}'.format(num))
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet' + os.path.sep

# Set steady state flag
steady_state_enable=False
# Set control flag 
control_enable=True
# Set mismatch flag
is_mismatch = True
mismatchType = 'H_'

r2_dB = torch.arange(-10,20,5) 
# r2_dB = torch.tensor([0]) 
q2_dB = r2_dB # Align both dynamic and observation noises
nExperiments = len(r2_dB)
r2 = 10**(r2_dB/10)
q2 = 10**(q2_dB/10)
# qopt = torch.sqrt(q2)

# Set array of loss values for plot
MB_LQG_avg_loss_dB = torch.zeros(1, nExperiments)
LQGNet_avg_loss_dB = torch.zeros(1, nExperiments)

for index in range(0,nExperiments):
   ####################
   ### Design Model ###
   ####################
   
   print("1/r2 [dB]: ", 10 * torch.log10(1/r2[index]))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q2[index]))

   # True model
   Q_true = (q2[index]) * torch.eye(m)
   R_true = (r2[index]) * torch.eye(n)
   sys_model = SystemModel(f, G, Q_true, h, R_true, T, T_test, is_mismatch=is_mismatch)
   sys_model.InitSequence(m1x_0, m2x_0)
   # Initalize LQR requlator matrices
   sys_model.InitCostMatrices(QT, Qx, Qu)
   # Compute LQR gains for finite and infinite time horizons
   sys_model.ComputeLQRgains()
   
   # # Mismatched model
   # sys_model_partial = SystemModel(fInacc, Q_true, h, R_true, T, T_test)
   # sys_model_partial.InitSequence(m1x_0, m2x_0)

   #####################################
   ### Noise Loader (Generate Noise) ###
   #####################################
   dataFolderName = 'Simulations' + os.path.sep + 'Toy_problems' + os.path.sep
   # noiseFileName = "Noise_T{:,.0f}_r2_{:,.3f}dB_q2_{:,.3f}dB".format(T,r2.cpu().detach().numpy()[0],q2.cpu().detach().numpy()[0])
   
   if r2_dB.cpu().detach().numpy()[index] < 0:
      simUseCase = "Linear_Dynamics_T{:,.0f}_r2_negative_{:,.0f}dB".format(T_test,r2_dB[index].abs().cpu().detach().numpy())
      
   elif r2_dB.cpu().detach().numpy()[index] == 0:
      simUseCase = "Linear_Dynamics_T{:,.0f}_r2_{:,.0f}dB".format(T_test,r2_dB[index].abs().cpu().detach().numpy())
      
   else: 
      simUseCase = "Linear_Dynamics_T{:,.0f}_r2_positive_{:,.0f}dB".format(T_test,r2_dB[index].abs().cpu().detach().numpy())
      
   
   # Set Noise File Name
   noiseFileName = "Noise_" + simUseCase
   # noiseFileName = noiseFileName.replace(".","_") + '.pt'
   noiseFilePath = dataFolderName + noiseFileName + '.pt'
   
   if not os.path.exists(noiseFilePath):
      print("Start Noise Gen")
      NoiseGen(sys_model, noiseFilePath, N_train, N_val, N_test)
   
   # Re-define sample size for training/validation/testing
   N_train = 1000
   N_val = 100
   N_test = T_test
   
   print("Noise Loader to GPU")
   [training_noise, validation_noise, test_noise] = NoiseLoader_GPU(noiseFilePath, N_test, N_val, N_train)
   
   # ###################################
   # ### Data Loader (Generate Data) ###
   # ###################################
   # if is_mismatch:
   #    dataFileName = "Data_MM_" + simUseCase
   # else:
   #    dataFileName = "Data_" + simUseCase
   
   # # dataFileName = dataFileName.replace(".","_") + '.pt'
   # dataFilePath = dataFolderName + dataFileName + '.pt'
   # if not os.path.exists(dataFilePath):
   #    print("Start Data Gen")
   #    DataGen(sys_model, dataFilePath, training_noise, validation_noise, test_noise, randomInit=False, steady_state=steady_state_enable, is_control_enable=control_enable)
      
 
   # print("Data Load")
   # [train_y, train_x, val_y, val_x, test_y, test_x] = DataLoader_GPU(dataFilePath)
   # # print("trainset size:",train_target.size())
   # # print("cvset size:",cv_target.size())
   # # print("testset size:",test_target.size())



   ################################
   ### Evaluate EKF, UKF and PF ###
   ################################
   
   print("Evaluate Kalman Filter True")
   
#    Q_search = (qopt[index]**2) * torch.eye(m)
#    sys_model = SystemModel(f, Q_search, h, R_true, T, T_test, m, n,"Toy")
#    sys_model.InitSequence(m1x_0, m2x_0)

#    sys_model_partial = SystemModel(fInacc, Q_search, h, R_true, T, T_test, m, n,"Toy")
#    sys_model_partial.InitSequence(m1x_0, m2x_0)
#    print("Evaluate Kalman Filter True")
#    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
#    print("Evaluate Kalman Filter Partial")
#    [MSE_KF_linear_arr_partial, MSE_KF_linear_avg_partial, MSE_KF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partial, test_input, test_target)

#    print("Evaluate UKF True")
#    [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(sys_model, test_input, test_target)
#    print("Evaluate UKF Partial")
#    [MSE_UKF_linear_arr_partial, MSE_UKF_linear_avg_partial, MSE_UKF_dB_avg_partial, UKF_out_partial] = UKFTest(sys_model_partial, test_input, test_target)
  
#    print("Evaluate PF True")
#    [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(sys_model, test_input, test_target)
#    print("Evaluate PF Partial")
#    [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial] = PFTest(sys_model_partial, test_input, test_target)


   # DatafolderName = 'Data' + '/'
   # DataResultName = '10x10_Ttest1000' 
   # torch.save({
   #             'MSE_KF_linear_arr': MSE_KF_linear_arr,
   #             'MSE_KF_dB_avg': MSE_KF_dB_avg,
   #             }, DatafolderName+DataResultName)

   ##################
   ###  KalmanNet ###
   ##################
   
   # Get model path
   if is_mismatch:
      modelFolder = 'KNet' + os.path.sep
      modelName = "KNet_MM_" + mismatchType + simUseCase
   else:
      modelFolder = 'KNet' + os.path.sep + 'Linear_SS_No_MM' + os.path.sep
      modelName = "KNet_" + simUseCase   
   
   # Run MB LQG and LGQNet
   if os.path.exists(modelFolder + "pipeline_" + modelName + ".pt"):
      KNet_Pipeline = torch.load(modelFolder + "pipeline_" + modelName + ".pt")
      # [MSE_EKF_dB_avg, MSE_EKF_dB_std, LQG_cost_dB, EKF_x_hat, EKF_x_true] = EKFTest(sys_model, test_noise, steady_state=steady_state_enable, is_control_enable=control_enable, EKF_enable=True)
      # [LQG_loss_summary, MSE_loss_total_summary, MSE_loss_position_summary] = KNet_Pipeline.NNTest(test_noise)
      
   else:
      torch.manual_seed(num)
      # Run LQG problem with EKF enable
      [MSE_EKF_dB_avg, MSE_EKF_dB_std, LQG_cost_dB, EKF_x_hat, EKF_x_true] = EKFTest(sys_model, test_noise, steady_state=steady_state_enable, is_control_enable=control_enable, EKF_enable=True)
      # Run LQR problem
      [_, _, LQR_cost_dB, FS_EKF_x_hat, FS_EKF_x_true] = EKFTest(sys_model, test_noise, steady_state=steady_state_enable, is_control_enable=control_enable, EKF_enable=False)
      
      print("KNet with full model info")
      KNet_Pipeline = Pipeline_EKF(strTime, modelFolder, modelName)
      KNet_Pipeline.LQR_ref_cost = LQR_cost_dB
      KNet_Pipeline.LQG_ref_cost = LQG_cost_dB
      KNet_Pipeline.MSE_EKF_ref_dB = MSE_EKF_dB_avg
      KNet_Pipeline.setssModel(sys_model)
      KNet_model = KalmanNetNN()
      KNet_model.Build(sys_model, steady_state=steady_state_enable, is_control_enable=control_enable)
      KNet_Pipeline.setModel(KNet_model)
      # KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=16, learningRate=0.5e-3, weightDecay=1e-3, alpha=0.0, beta=1.0)
      KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=16, learningRate=0.5e-3, weightDecay=1e-3, alpha=0.0, beta=1.0)


      KNet_Pipeline.NNTrain(training_noise, validation_noise, num_restarts=2)
      [LQG_loss_summary, MSE_loss_total_summary, MSE_loss_position_summary] = KNet_Pipeline.NNTest(test_noise)
      KNet_Pipeline.save()
   
# Set array of LQG + MSE loss values for plot
MB_LQG_avg_loss_dB = torch.zeros(1, nExperiments)
LQGNet_avg_loss_dB = torch.zeros(1, nExperiments)

MB_MSE_avg_loss_dB = torch.zeros(1, nExperiments)
LQGNet_MSE_loss_dB = torch.zeros(1, nExperiments)

# Set array of MM SS LGQ + MSE loss values for plot
MM_MB_LQG_avg_loss_dB = torch.zeros(1, nExperiments)
MM_LQGNet_avg_loss_dB = torch.zeros(1, nExperiments)

MM_MSE_avg_loss_dB = torch.zeros(1, nExperiments)
MM_LQGNet_MSE_loss_dB = torch.zeros(1, nExperiments)


# Collect LQG losses vs KF+LQR
for i in range(nExperiments):
   if r2_dB.cpu().detach().numpy()[i] < 0:
      simUseCase = "Linear_Dynamics_T{:,.0f}_v_negative_{:,.0f}dB".format(T_test,r2_dB[i].abs().cpu().detach().numpy())
      
   elif r2_dB.cpu().detach().numpy()[i] == 0:
      simUseCase = "Linear_Dynamics_T{:,.0f}_v_{:,.0f}dB".format(T_test,r2_dB[i].abs().cpu().detach().numpy())
      
   else: 
      simUseCase = "Linear_Dynamics_T{:,.0f}_v_positive_{:,.0f}dB".format(T_test,r2_dB[i].abs().cpu().detach().numpy())
      
   # Where to save the new model
   modelFolder = 'KNet' + os.path.sep + 'Linear_SS_No_MM' + os.path.sep
   modelName = "KNet_" + simUseCase   

   KNet_Pipeline = torch.load(modelFolder + "pipeline_" + modelName + ".pt")
   # Save LQG loss for current experiment
   MB_LQG_avg_loss_dB[0,i] = KNet_Pipeline.LQG_ref_cost
   LQGNet_avg_loss_dB[0,i] = KNet_Pipeline.LQR_test_dB_avg
   MB_MSE_avg_loss_dB[0,i] = KNet_Pipeline.MSE_EKF_ref_dB
   LQGNet_MSE_loss_dB[0,i] = KNet_Pipeline.MSE_test_dB_avg
   # Save LQG loss for mismatch dynamics
   modelFolder = 'KNet' + os.path.sep + os.path.sep
   modelName = "KNet_MM_" + simUseCase   
   KNet_Pipeline = torch.load(modelFolder + "pipeline_" + modelName + ".pt")
   MM_MB_LQG_avg_loss_dB[0,i] = KNet_Pipeline.LQG_ref_cost
   MM_LQGNet_avg_loss_dB[0,i] = KNet_Pipeline.LQR_test_dB_avg
   MM_MSE_avg_loss_dB[0,i] = KNet_Pipeline.MSE_EKF_ref_dB
   MM_LQGNet_MSE_loss_dB[0,i] = KNet_Pipeline.MSE_test_dB_avg


   
####################
### Plot results ###
####################
# KNet_Pipeline = torch.load(modelFolder+"pipeline_KNet_Linear_Dynamics_v_30dB.pt")
# p = Plot(KNet_Pipeline.folderName, KNet_Pipeline.modelName)
p = Plot(KNet_Pipeline)
p.plot_LQG_loss_vs_SNR(LQGNet_avg_loss_dB, MB_LQG_avg_loss_dB, LQGNet_MSE_loss_dB, MB_MSE_avg_loss_dB,
                       MM_LQGNet_avg_loss_dB, MM_MB_LQG_avg_loss_dB, MM_LQGNet_MSE_loss_dB, MM_MSE_avg_loss_dB, vdB)
   # p.plot_lqr_and_mse(-1.9, figSize=(25,20), lineWidth=3, ylim2=[8.5,25], ylim3=[-2.5,16], fontSize=26)

   # # KNet with model mismatch
   # ## Build Neural Network
   # print("KNet with partial model info")
   # KNet_model = KalmanNetNN()
   # KNet_model.Build(sys_model_partial)
   # # Model = torch.load('KNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
   # ## Train Neural Network
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet_partial")
   # KNet_Pipeline.setssModel(sys_model_partial)
   # KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
   # KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
   # ## Test Neural Network
   # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
   # KNet_Pipeline.save()