import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import os
import datetime
def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    os.makedirs('outputs', exist_ok=True)
    now = datetime.datetime.now()
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'outputs/model_{now.strftime("%Y-%m-%d_%H-%M")}.pth')
    
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    os.makedirs('outputs', exist_ok=True)
    now = datetime.datetime.now()
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'outputs/accuracy_{now.strftime("%Y-%m-%d_%H-%M")}.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/loss_{now.strftime("%Y-%m-%d_%H-%M")}.png')