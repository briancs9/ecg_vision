from torchvision.transforms import v2
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import torch
import pandas as pd
import config
from wfdb.processing import normalize_bound
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

config = config.Config()

## config parameters
BATCH_SIZE = config.batch_size
TRAIN_DATA_PATH = config.train_data_path
TEST_DATA_PATH = config.test_data_path


transform_img = v2.Compose([
    v2.Lambda(lambda x: normalize_bound(x.iloc[:,0:12])),
    v2.Lambda(lambda x: torch.tensor(x.values, dtype=torch.float32).unsqueeze(0)),
    v2.Resize(size=(5000,12), antialias=True),
    v2.Lambda(lambda x: x.permute(0, 2, 1).squeeze(0))
])

train_dataset = DatasetFolder(root=TRAIN_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")
test_dataset = DatasetFolder(root=TEST_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

