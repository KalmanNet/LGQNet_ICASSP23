import torch
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

#######################
### Size of DataSet ###
#######################

# # Number of Training Examples
# N_E = 1000

# # Number of Cross Validation Examples
# N_CV = 10

# N_T = 200

# Sequence Length
# T = 20
# T_test = 20

# Train/Validation/Test data size
N_train = 10000
N_val = 2000
N_test = 2000

#################
## Design #10 ###
#################
F10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

############
## 2 x 2 ###
############
# m = 2
# n = 2
# F = F10[0:m, 0:m]
# H = torch.eye(2)
# m1_0 = torch.tensor([[0.0], [0.0]]).to(cuda0)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)


#############
### 5 x 5 ###
#############
# m = 5
# n = 5
# F = F10[0:m, 0:m]
# H = H10[0:n, 10-m:10]
# m1_0 = torch.zeros(m, 1).to(cuda0)
# # m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(cuda0)
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)

##############
## 10 x 10 ###
##############
# m = 10
# n = 10
# F = F10[0:m, 0:m]
# H = H10
# m1_0 = torch.zeros(m, 1).to(cuda0)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)

def DataGen_True(SysModel_data, fileName, T):

    SysModel_data.GenerateBatch(1, T, randomInit=False)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    # torch.save({"True Traj":[test_target],
    #             "Obs":[test_input]},fileName)
    torch.save([test_input, test_target], fileName)

def DataGen(SysModel_data, fileName, training_noise, validation_noise, test_noise, randomInit=False, steady_state=False, is_control_enable=True):

    T = SysModel_data.T
    T_test = SysModel_data.T_test
    N_train = training_noise[0].size()[0]
    N_val = validation_noise[0].size()[0]
    N_test = test_noise[0].size()[0]
    
    ##################################
    ### Generate Training Sequence ###
    ##################################
    # Get Process/Observation Noise
    Q_noise, R_noise = training_noise
    SysModel_data.GenerateBatch(N_train, T, Q_noise, R_noise, randomInit=randomInit, seqInit=False, T_test=0, steady_state=False, is_control_enable=True)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    # Get Process/Observation Noise
    Q_noise, R_noise = validation_noise
    SysModel_data.GenerateBatch(N_val, T, Q_noise, R_noise, randomInit=randomInit, seqInit=False, T_test=0, steady_state=False, is_control_enable=True)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    # Get Process/Observation Noise
    Q_noise, R_noise = test_noise
    SysModel_data.GenerateBatch(N_test, T_test, Q_noise, R_noise, randomInit=randomInit, seqInit=False, T_test=0, steady_state=False, is_control_enable=True)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):

    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DataLoader_GPU(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName),pin_memory = False)
    training_input = training_input.squeeze().to(dev)
    training_target = training_target.squeeze().to(dev)
    cv_input = cv_input.squeeze().to(dev)
    cv_target =cv_target.squeeze().to(dev)
    test_input = test_input.squeeze().to(dev)
    test_target = test_target.squeeze().to(dev)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):
    data_target = list(torch.split(data_target,T,2))
    data_input = list(torch.split(data_input,T,2))
    data_target.pop()
    data_input.pop()
    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))
    return [data_target, data_input]

def NoiseGen(SysModel_data, fileName, N_train, N_val, N_test):
    # Trajectory length 
    T = SysModel_data.T_test
    # Gen training noise sequence
    training_noise = SysModel_data.GenNoiseSequence(T, N_train)
    # Gen cross validation noise sequence
    validation_noise = SysModel_data.GenNoiseSequence(T, N_val)
    # Gen test noise sequence
    test_noise = SysModel_data.GenNoiseSequence(T, N_test)
    
    #################
    ### Save Data ###
    #################
    torch.save((training_noise, validation_noise, test_noise), fileName)
    
def NoiseLoader_GPU(fileName, N_test, N_val, N_train):
    
    [training_noise, validation_noise, test_noise] = torch.load(fileName) #torch.utils.data.DataLoader(torch.load(fileName),pin_memory = False)
    
    if torch.cuda.is_available():
        training_noise      = [D[:N_train].to("cuda:0") for D in training_noise]
        validation_noise    = [D[:N_val].to("cuda:0") for D in validation_noise]
        test_noise          = [D[:N_test].to("cuda:0") for D in test_noise]
    else:
        training_noise      = [D[:N_train] for D in training_noise]
        validation_noise    = [D[:N_val] for D in validation_noise]
        test_noise          = [D[:N_test] for D in test_noise]
    
    return [training_noise, validation_noise, test_noise]