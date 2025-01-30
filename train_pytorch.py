import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
import models
from utils import save_model, save_plots
import ecg_model


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
    help='number of epochs to train our network for')
args = vars(parser.parse_args())


# learning_parameters 
lr = 1e-3
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
model = ecg_model.CNNTransformerHybrid(num_classes=1).to(device)

print(model)


# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss function
criterion = nn.BCEWithLogitsLoss()

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
    tv.transforms.Lambda(lambda x: torch.tensor(x.values, dtype=torch.float32)),
    tv.transforms.Lambda(lambda x: x.transpose(0, 1)),  # Now [8, Length]
    tv.transforms.Resize((8,5000), antialias=True),
    #tv.transforms.Lambda(lambda x: x.unsqueeze(0)) 
])

train_dataset = tv.datasets.DatasetFolder(root=TRAIN_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")
test_dataset = tv.datasets.DatasetFolder(root=TEST_DATA_PATH, transform=transform_img, loader=lambda x: pd.read_csv(x, sep=",", header=0), extensions=".csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




# training validation loops
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        # forward pass
        outputs = model(image).squeeze(1).float()
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        train_running_correct += (outputs.round() == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device).float()
            # forward pass
            outputs = model(image).squeeze(1).float()
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            #_, preds = torch.max(outputs.data, 1)
            valid_running_correct += (outputs.round() == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                 criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    time.sleep(5)
    
# save the trained model weights
save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')