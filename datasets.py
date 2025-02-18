from torchvision.transforms import v2
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import torch
import pandas as pd
import config
from wfdb.processing import normalize_bound
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

config = config.Config()

## config parameters
BATCH_SIZE = config.batch_size
TRAIN_DATA_PATH = config.train_data_path
TEST_DATA_PATH = config.test_data_path


if config.transformer_model == 'transformer':
    transform_img = v2.Compose([
        v2.Lambda(lambda x: normalize_bound(x.iloc[:,0:12])),
        v2.Lambda(lambda x: torch.tensor(x.values, dtype=torch.float32).unsqueeze(0)),
        v2.Resize(size=(5000,12), antialias=True),
        v2.Lambda(lambda x: x.squeeze(0))
    ])
elif config.transformer_model == 'conv':
    transform_img = v2.Compose([
        v2.Lambda(lambda x: x.iloc[:,0:12]),
        v2.Lambda(lambda x: normalize_bound(x)),
        v2.Lambda(lambda x: torch.tensor(x.values.astype(np.float32), dtype=torch.float32)),
        v2.Lambda(lambda x: x.unsqueeze(0)),
        v2.Resize(size=(5000,12), antialias=True),
        v2.Lambda(lambda x: x.squeeze(0))
    ])
else:
    raise ValueError(f"Invalid model: {config.transformer_model}")


train_dataset = DatasetFolder(root=TRAIN_DATA_PATH, 
                              transform=transform_img, 
                              loader=lambda x: pd.read_csv(x, sep=",", header=0), 
                              extensions=".csv")
test_dataset = DatasetFolder(root=TEST_DATA_PATH, 
                             transform=transform_img, 
                             loader=lambda x: pd.read_csv(x, sep=",", header=0), 
                             extensions=".csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

