import torch
import os
import logging

def set_device_name(name): 
    os.environ['DEVICE'] = name
    
def get_device_name(): 
    return os.environ['DEVICE'] if 'DEVICE' in os.environ else None

def get_device(i=0):
    name = get_device_name()

    if 'gpu' in name:
        name = f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else 'cpu'
 
    return torch.device(name)

        
def dict_join(a, b):
    a.copy()
    a.update(b)
    return a