import torchvision as tv

from torch.utils.data import DataLoader
import torch
import pandas as pd
import torch.nn as nn

# batch size
BATCH_SIZE = 32
TRAIN_DATA_PATH = 'data/train'
TEST_DATA_PATH = 'data/test'

class UnsqueezeImage(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(0)
        return x   

transform_img = tv.transforms.Compose([
    lambda x: torch.tensor(x.values, dtype=torch.float32),
    UnsqueezeImage() # c, h, w
])

train_dataset = tv.datasets.DatasetFolder(root=TRAIN_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")
test_dataset = tv.datasets.DatasetFolder(root=TEST_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

