import torch
import pandas as pd
from utils import get_device

def df_to_tensor(data_frame, device=None):  
    device = device if device else get_device()  
    tensor = torch.tensor(data_frame.values, device=device)
    return tensor.type(torch.cuda.FloatTensor if 'cuda' in device.type else torch.FloatTensor)

def to_single_col_df(values, col='any'): return pd.DataFrame({col: values})
