import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import numpy as np


class TransformerModel(nn.Module):
    def __init__(self, d_output: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.01, pos_encoder: str = 'dummy', activation: str = "gelu"):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.d_output = d_output
        if pos_encoder=='dummy':
           self.pos_encoder = DummyPositionalEncoding(d_model)
        elif pos_encoder=='t2v':
           self.pos_encoder = t2vSineActivation(1, d_model) 
        else:
            raise Exception('Positional encoder must be dummy or t2v')
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, norm_first=True, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear_output = nn.Linear(d_model, d_output)
        self.init_weights()

    def init_weights(self) -> None:
        self.linear_output.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.linear_output.weight)

    def forward(self, src: Tensor, stamp: Tensor, src_mask: Tensor = None) -> Tensor:
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src, stamp)

        output = self.transformer_encoder(src, src_mask)
        output = self.linear_output(output)

        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz,sz) * float('-inf'), diagonal=1)
        

##### POSITIONAL ENCODINGS #####
class DummyPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        

    def forward(self, x: Tensor, stamp: Tensor) -> Tensor:
        """
        Argumenrs:
            x: Tensor, shape ``[seq_len, batch_size, 1]``
        """
        stamp_init = stamp[:,0].unsqueeze(1)
        stamp_final = stamp[:,-1].unsqueeze(1)

        T_aux = stamp_final - stamp_init

        aux = (stamp-stamp_init)/T_aux

        device = x.device
        pe = torch.zeros(stamp.shape[0], stamp.shape[1], 2).to(device)
        pe[:, :, 0] = torch.sin(np.pi*aux)
        pe[:, :, 1] = torch.cos(np.pi*aux)

        x = torch.cat((x, pe[:x.size(0)]), 2)
        
        return x
    
    
class t2vSineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(t2vSineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, input, stamp) -> Tensor:
        ### some kind of normalization is needed (dataset values are very large). this is probably not the best way to do it
        ### TODO
        ### could probalby normalize the full dataset to a single initial value
        stamp_init = stamp[:,0].view(stamp.shape[0],1)
        stamp_final = stamp[:,-1].view(stamp.shape[0],1)
        T_aux = stamp_final - stamp_init
        aux = (stamp-stamp_init)/T_aux

        v1 = torch.matmul(aux, self.w0) + self.b0
        v2 = self.f(torch.matmul(stamp, self.w) + self.b)
        return torch.cat([input, v1, v2], -1)