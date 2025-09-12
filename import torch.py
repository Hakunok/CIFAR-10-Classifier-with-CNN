import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

# Data Class Definition
class ImageDataSet(Dataset):
  def __init__(self):
    self.features = torch.randn(60000, 32, 32, 3) # 60000 images that are 32x32 with 3 color channels
    self.labels = torch.randint(0, 2, (60000,)) # labels are either 0 or 1

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx): # getitem requires the idx parameter
    self.features[idx].clamp_(0, 1)
    return (self.features[idx], self.labels[idx])
  

