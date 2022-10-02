import torch

def mean_and_std_linear_and_dB(arr):
    mean = torch.mean(arr)
    mean_dB = 10 * torch.log10(mean)

    std = torch.std(arr, unbiased=True)
    std_dB = 10 * torch.log10(std + mean) - mean_dB

    return mean, mean_dB, std, std_dB