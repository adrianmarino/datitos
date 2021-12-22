import torch

def try_gpu(i=0):
    return torch.device('cpu')
    # return torch.device(f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else 'cpu')


def dict_join(a, b):
    a.copy()
    a.update(b)
    return a