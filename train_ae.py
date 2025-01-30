import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
import models
from utils import save_model
from torch.utils.data import DataLoader
import pandas as pd
import torchvision as tv
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()
from wfdb.processing import normalize_bound



# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=1,
    help='number of epochs to train our network for')
args = vars(parser.parse_args())



# learning_parameters 
lr = 1e-3
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'mps')
print(f"Computation device: {device}\n")

model = models.ECG_Autoencoder().to(device)

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss function
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# batch size
BATCH_SIZE = 64
TRAIN_DATA_PATH = 'all_muse_records/csv_records/'

class UnsqueezeImage(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(0)
        return x

transform_img = tv.transforms.Compose([
    tv.transforms.Lambda(lambda x: torch.tensor(normalize_bound(x.iloc[:,[0,1,6,7,8,9,10,11]].values, lb=0, ub=1), dtype=torch.float32)),
    UnsqueezeImage(),
    tv.transforms.Resize((5000,8), antialias=True),
])

train_dataset = tv.datasets.DatasetFolder(root=TRAIN_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)




# training validation loops
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, _ = data
        image = image.to(device)
        optimizer.zero_grad()
        outputs = model(image)[0]
        loss = criterion(outputs, image)
        loss.backward()
        optimizer.step()
        log.record(pos=(epoch + (i+1)/len(trainloader)), loss=loss.item(), end='\r')

    return losses


# start the training
for epoch in range(epochs):
    train_losses = []
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_loss = train(model, train_loader, optimizer, criterion)
    scheduler.step(train_loss)
    print('-'*50)
    time.sleep(5)
    
# save the trained model weights
save_model(epochs, model, optimizer, criterion)

# loss plots

print('TRAINING COMPLETE')