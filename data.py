import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


def get_columns(dataframe, features, fields):
  aux_list = [False for i in dataframe]
  aux_list_fields = [False for i in dataframe]

  for i in features:
    aux_list |= dataframe.columns.get_level_values(0)==i
  
  for i in fields:
    aux_list_fields |= dataframe.columns.get_level_values(1)==i

  return aux_list & aux_list_fields


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, fields, stamp, sequence_length = 5, device = 'cpu'):
        self.features = features
        self.fields = fields
        self.target = target
        self.sequence_length = sequence_length

        columns = get_columns(dataframe, self.features, self.fields)
        self.y = torch.tensor(dataframe[target][fields].values).float().to(device) # device?
        self.x = torch.tensor(dataframe.iloc[:,columns].values).float().to(device)
        self.stamp = torch.tensor(dataframe[target][stamp].values).squeeze().to(device)


    def __len__(self):
        if(self.sequence_length == -1):
            return self.x.shape[0]
        
        return self.x.shape[0] - self.sequence_length

    def __getitem__(self, i):
        data = self.x[i:self.sequence_length+i]
        output = self.y[i:self.sequence_length+i]
        stamp = self.stamp[i:self.sequence_length+i]

        return data, stamp, output

        