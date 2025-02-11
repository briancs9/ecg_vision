import config
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
import ecg_conv_models as models
from utils import save_model, save_plots
import torchvision as tv
import datasets
import ecg_transformers as transformers

config = config.Config()

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=config.epochs,
    help='number of epochs to train our network for')
args = vars(parser.parse_args())


# learning_parameters 
lr = config.learning_rate
epochs = args['epochs']
device = config.device

print(f"Computation device: {device}\n")


if config.transformer_model == 'conv':
    model = models.ECG_Model()
elif config.transformer_model == 'transformer':
    model = transformers.CCT(num_classes=config.num_classes, 
                        num_heads=config.num_heads, 
                        num_transformer_layers=config.num_transformer_layers, 
                        d_model=config.d_model, 
                        seq_pool=config.seq_pool)
else:
    raise ValueError(f"Invalid model: {config.transformer_model}")

print(model)


# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

## define training parameters
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = config.criterion
BATCH_SIZE = config.batch_size
TRAIN_DATA_PATH = config.train_data_path
TEST_DATA_PATH = config.test_data_path


train_loader = datasets.train_loader
valid_loader = datasets.valid_loader




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
    
# save the trained model weights
save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_loss)

print('TRAINING COMPLETE')