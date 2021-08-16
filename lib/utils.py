import torch

def try_gpu(i=0):
    return torch.device(f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else 'cpu')

