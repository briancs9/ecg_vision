from torchvision.transforms import v2
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import pandas as pd
import config
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore", category=FutureWarning)

config = config.Config()

## config parameters
BATCH_SIZE = config.batch_size


def column_wise_normalize(x):
    """
    Normalize each column independently to the range [0, 1] using min-max normalization.
    Each column is normalized based on its own min and max values.
    
    Args:
        x: Input tensor of shape (H, W) or (1, H, W)
    
    Returns:
        Normalized tensor of the same shape as input, with values in [0, 1] per column
    """
    # Handle different input shapes
    original_shape = x.shape
    squeeze_first = False
    
    if len(original_shape) == 3 and original_shape[0] == 1:
        # Shape is (1, H, W) - we'll squeeze and unsqueeze
        x = x.squeeze(0)
        squeeze_first = True
    
    # x is now (H, W)
    H, W = x.shape
    
    # Compute min and max along the first dimension (rows) for each column
    # min_vals and max_vals will have shape (W,)
    min_vals = x.min(dim=0, keepdim=True)[0]  # Shape: (1, W)
    max_vals = x.max(dim=0, keepdim=True)[0]  # Shape: (1, W)
    
    # Handle edge case where max == min (constant column)
    # Add small epsilon to avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    
    # Normalize: (x - min) / (max - min)
    x_normalized = (x - min_vals) / range_vals
    
    # Restore original shape if needed
    if squeeze_first:
        x_normalized = x_normalized.unsqueeze(0)
    
    return x_normalized


def column_wise_resize(x, target_rows=5000):
    """
    Resize tensor to target_rows by averaging/interpolating along each column individually.
    Only operates on the row dimension (first dimension), preserving column structure.
    
    Handles images of any size:
    - If input is already target_rows: returns unchanged
    - If input is smaller (e.g., 2500 rows): upsamples to target_rows using linear interpolation
    - If input is larger: downsamples to target_rows using linear interpolation
    
    This function is designed to work with PyTorch DataLoader, where transforms are
    applied per-sample before batching.
    
    Args:
        x: Input tensor of shape (H, W) or (1, H, W) where H can be any size
        target_rows: Target number of rows (default: 5000)
    
    Returns:
        Resized tensor of shape (target_rows, W) or (1, target_rows, W)
    """
    # Handle different input shapes
    original_shape = x.shape
    squeeze_first = False
    
    if len(original_shape) == 3 and original_shape[0] == 1:
        # Shape is (1, H, W) - we'll squeeze and unsqueeze
        x = x.squeeze(0)
        squeeze_first = True
    
    # x is now (H, W)
    H, W = x.shape
    
    # If already at target size, return as is (no processing needed)
    if H == target_rows:
        if squeeze_first:
            return x.unsqueeze(0)
        return x
    
    # For any other size (2500, or any other), resize using interpolation
    # Transpose to (W, H) so we can interpolate along the last dimension
    # This allows us to interpolate each column independently
    x = x.transpose(0, 1)  # Now (W, H)
    
    # Add batch dimension for 1D interpolation: (W, H) -> (1, W, H)
    # F.interpolate with mode='linear' expects (N, C, L) where N=batch, C=channels, L=length
    x = x.unsqueeze(0)  # Now (1, W, H)
    
    # Use linear interpolation along the last dimension (height/rows)
    # mode='linear' with align_corners=False will average adjacent values
    # This works for both upsampling (2500 -> 5000) and downsampling (if needed)
    x = torch.nn.functional.interpolate(x, size=target_rows, mode='linear', align_corners=False)
    
    # Remove batch dimension: (1, W, target_rows) -> (W, target_rows)
    x = x.squeeze(0)
    
    # Transpose back: (W, target_rows) -> (target_rows, W)
    x = x.transpose(0, 1)
    
    # Restore original shape if needed
    if squeeze_first:
        x = x.unsqueeze(0)
    
    return x


if config.transformer_model == 'transformer':
    transform_img = v2.Compose([
        v2.Lambda(lambda x: x.iloc[:,0:8]),
        v2.Lambda(lambda x: torch.tensor(x.values, dtype=torch.float32).unsqueeze(0)),
        v2.Lambda(lambda x: column_wise_resize(x, target_rows=5000)),
        v2.Lambda(lambda x: column_wise_normalize(x))
    ])
elif config.transformer_model == 'conv':
    transform_img = v2.Compose([
        v2.Lambda(lambda x: x.iloc[:,0:8].values),
        v2.Lambda(lambda x: torch.tensor(x.astype(np.float32), dtype=torch.float32)),
        v2.Lambda(lambda x: x.unsqueeze(0)),
        v2.Lambda(lambda x: column_wise_resize(x, target_rows=5000)),
        v2.Lambda(lambda x: column_wise_normalize(x)),
        v2.Lambda(lambda x: x.squeeze(0))
    ])
else:
    raise ValueError(f"Invalid model: {config.transformer_model}")


class ECGDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = pd.read_csv(img_path, sep=",", header=0)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = ECGDataset(annotations_file=config.train_annotations_file, 
                           img_dir=config.data_path, 
                           transform=transform_img)
val_dataset = ECGDataset(annotations_file=config.val_annotations_file, 
                          img_dir=config.data_path, 
                          transform=transform_img)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
