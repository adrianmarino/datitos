# Pytorch...
import torch
from torch.optim import Adam, SGD
from torch.nn    import Sequential, \
                        Softmax, \
                        LeakyReLU

from model  import CommonModel, \
                   softmax_cross_entropy_fn, \
                   softmax_pred_out, \
                   dense_stack

from utils import get_device

class FifaModel1(CommonModel):
    def __init__(
        self, 
        units_per_layer, 
        lr,
        momentum,
        dropout, 
        negative_slope
    ):
        model = Sequential(*dense_stack(
            units_per_layer, 
            LeakyReLU(negative_slope=negative_slope), dropout), 
            Softmax(dim=1)
        )

        model = model.to(device=get_device())

        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

        super().__init__(model, softmax_cross_entropy_fn, optimizer, softmax_pred_out)