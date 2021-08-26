import torch
import torchvision
from torch.utils.data import Dataset


class LinearNoisyData(Dataset):

    def __init__(self):
        self.num_samples = 500
        self.weights = torch.tensor([4.0, 5.0])
        self.x = torch.randn((500, 2))
        self.y = torch.sum(self.weights * self.x, dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MNIST(Dataset):
    pass
