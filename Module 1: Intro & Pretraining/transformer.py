import torch

class PositionalEncoding:
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        # shape of pe: [1, max_length, d_model]
        # Why we don't need this:
        # self.d_model = d_model
        # self.max_length = max_length
        pe = torch.zeros(shape=(1,max_length, d_model))
        position = torch.arange(0,max_length)
        div_term = (10_000**(torch.arange(0, d_model, 2)/d_model))
        pe[:,:,]
        


        

    

class MLP:
    ...

class Normalization:
    ...

class SelfAttention:
    ...

class Encoder:
    ...
