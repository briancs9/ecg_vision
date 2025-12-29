import config as config_module
import torch
import argparse
import torch.optim as optim
from tqdm.auto import tqdm
import models
from utils import save_model, save_plots, create_model
import datasets
import json
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

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

# Initialize model based on config
model = create_model(config, models)

model.to(device)

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
# Custom scheduler with warmup and linear annealing
def lr_lambda(epoch):
    initial_lr = config.learning_rate
    warmup_epochs = config.warmup
    annealing_rate = config.annealing
    total_epochs = epochs
    
    if epoch < warmup_epochs:
        # Warmup phase: linearly increase from 0 to initial_lr
        return epoch / warmup_epochs
    else:
        # Linear annealing phase: decay from initial_lr to initial_lr * annealing_rate
        # Calculate progress through annealing phase (0 to 1)
        annealing_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        # Linearly interpolate between 1.0 and annealing_rate
        return 1.0 - annealing_progress * (1.0 - annealing_rate)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



# TRAINING LOOP
def train(model, trainloader, optimizer, criterion):
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
    all_probs = []
    all_labels = []
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
            # collect probabilities and labels for AUPRC calculation
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / num_batches
    epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))
    # calculate AUPRC
    auprc = average_precision_score(all_labels, all_probs)
    # calculate F1 score (convert probabilities to binary predictions)
    all_probs_array = np.array(all_probs)
    all_labels_array = np.array(all_labels)
    binary_preds = (all_probs_array >= 0.5).astype(int)
    f1 = f1_score(all_labels_array, binary_preds)
    return epoch_loss, epoch_acc, auprc, f1

# LIST TO KEEP TRACK OF LOSSES AND ACCURACIES
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
valid_auprc = []
valid_f1 = []

# START THE TRAINING
best_valid_acc = 0
best_f1_score = 0.0
best_valid_loss = float('inf')
# Pre-compute config_state once to avoid recreating it multiple times
config_state = {key:value for key, value in config.__dict__.items() if not key.startswith('_') and not callable(value)}
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc, valid_epoch_auprc, valid_epoch_f1 = validate(model, val_loader,  
                                                 criterion)
    # Get current learning rate
    updated_lr = scheduler.get_last_lr()
    # Step scheduler after each epoch
    scheduler.step()
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    valid_auprc.append(valid_epoch_auprc)
    valid_f1.append(valid_epoch_f1)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}, validation AUPRC: {valid_epoch_auprc:.4f}, validation F1: {valid_epoch_f1:.4f}")
    print(f"Updated learning rate for next epoch: {updated_lr[0]:.6f}")
    print('-'*50)
    
    # Save model if F1 score improved OR validation loss decreased
    should_save = False
    if valid_epoch_f1 > best_f1_score:
        best_f1_score = valid_epoch_f1
        should_save = True
        print(f"New best F1 score: {best_f1_score:.4f}")
    if valid_epoch_loss < best_valid_loss:
        best_valid_loss = valid_epoch_loss
        should_save = True
        print(f"New best validation loss: {best_valid_loss:.3f}")
    
    if should_save:
        save_model(epoch+1, model, optimizer, criterion, config_dict=config_state)

save_plots(train_acc, valid_acc, train_loss, valid_loss)

print('TRAINING COMPLETE')

