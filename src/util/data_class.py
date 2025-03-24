import torch


## Create Pytroch class instance for ADM dataset
class ADMDataSet(torch.utils.data.Dataset):
  def __init__(self, g, s):
    self.g = g
    self.s = s
    self.len = len(g)

  def __len__(self):
    return self.len

  def __getitem__(self, ind):
    return self.g[ind, :], self.s[ind, :]