import torch
from utils import try_gpu

def df_to_tensor(data_frame, device=try_gpu()):    
    tensor = torch.tensor(data_frame.values, device=device)
    return tensor.type(torch.cuda.FloatTensor if 'cuda' in device.type else torch.FloatTensor)