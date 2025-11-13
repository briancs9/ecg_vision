import config as config_module
import torch
import argparse
import torch.optim as optim
from tqdm.auto import tqdm
import models
from utils import save_model, save_plots
import datasets
import cct2 as transformers
import json

# CONSTRUCT THE ARGUMENT PARSER
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=None,
    help='number of epochs to train our network for')
parser.add_argument('-c', '--config', type=str, default=None,
    help='path to JSON config file to load parameters from')
args = vars(parser.parse_args())

# Initialize config, optionally loading from JSON file
config = config_module.Config(config_json_path=args['config'])

# Use command line epochs if provided, otherwise use config default
epochs = args['epochs'] if args['epochs'] is not None else config.epochs

# LEARNING PARAMETERS 
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

model.to(device)

print(f'{config.transformer_model} model initialized')

# TOTAL PARAMETERS AND TRAINABLE PARAMETERS
total_params = 0
total_trainable_params = 0
for p in model.parameters():
    num_params = p.numel()
    total_params += num_params
    if p.requires_grad:
        total_trainable_params += num_params
print(f"{total_params:,} total parameters.")
print(f"{total_trainable_params:,} training parameters.")

## TRAINING PARAMETERS
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
pos_weight = torch.tensor([config.pos_weight]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_loader = datasets.train_loader
val_loader = datasets.val_loader

## LEARNING RATE SCHEDULER
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                         max_lr=config.learning_rate,
                                         total_steps=total_steps,
                                         pct_start=0.1,  # 10% of training for warmup
                                         anneal_strategy='cos',
                                         div_factor=100.0,
                                         final_div_factor=1000.0)



# TRAINING LOOP
def train(model, trainloader, optimizer, criterion, scheduler):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    num_batches = len(trainloader)
    for data in tqdm(trainloader, total=num_batches):
        image, labels = data
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        # forward pass
        outputs = model(image).squeeze(1)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy (apply sigmoid to logits before rounding)
        probs = torch.sigmoid(outputs)
        train_running_correct += (probs.round() == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
        # step scheduler after each batch (for OneCycleLR)
        scheduler.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / num_batches
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# VALIDATION LOOP
def validate(model, val_loader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    num_batches = len(val_loader)
    with torch.no_grad():
        for data in tqdm(val_loader, total=num_batches):
            image, labels = data
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            # forward pass
            outputs = model(image).squeeze(1)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy (apply sigmoid to logits before rounding)
            probs = torch.sigmoid(outputs)
            valid_running_correct += (probs.round() == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / num_batches
    epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))
    return epoch_loss, epoch_acc

# LIST TO KEEP TRACK OF LOSSES AND ACCURACIES
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# START THE TRAINING
best_acc = 0
# Pre-compute config_state once to avoid recreating it multiple times
config_state = {key:value for key, value in config.__dict__.items() if not key.startswith('_') and not callable(value)}
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion, scheduler)
    valid_epoch_loss, valid_epoch_acc = validate(model, val_loader,  
                                                 criterion)
    # Get current learning rate (scheduler is stepped per batch, not per epoch)
    updated_lr = scheduler.get_last_lr()
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print(f"Updated learning rate for next epoch: {updated_lr[0]:.6f}")
    print('-'*50)
    
    if valid_epoch_acc > best_acc:
        best_acc = valid_epoch_acc
        save_model(epochs, model, optimizer, criterion, config_dict=config_state)

save_plots(train_acc, valid_acc, train_loss, valid_loss)

print('TRAINING COMPLETE')
