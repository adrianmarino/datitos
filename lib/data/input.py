from torch.utils import data
from data import df_to_tensor

class InputDataIteratorFactory:
    @staticmethod
    def create(features, target, batch_size=16, shuffle=True):
        dataset = data.TensorDataset(df_to_tensor(features), df_to_tensor(target))
        return data.DataLoader(dataset, batch_size, shuffle=shuffle)
