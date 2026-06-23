import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import os
import datetime
import json
import numpy as np
import config

config = config.Config()


def create_model(config, models_module):
    if config.transformer_model == 'transformer':
        # Calculate dim_feedforward from mlp_ratio if available
        dim_feedforward = getattr(config, 'd_model', 256) * getattr(config, 'mlp_ratio', 4)
        # Handle backward compatibility: convert use_pos_encoding to positional_embedding
        use_pos_encoding = getattr(config, 'use_pos_encoding', True)
        if hasattr(config, 'positional_embedding'):
            positional_embedding = config.positional_embedding
        else:
            positional_embedding = 'sine' if use_pos_encoding else 'none'
        model = models_module.ECG_Transformer(
            num_classes=config.num_classes,
            num_encoder_layers=getattr(config, 'num_transformer_layers', 6),
            d_model=getattr(config, 'd_model', 64),
            nhead=getattr(config, 'num_heads', 8),
            dim_feedforward=dim_feedforward,
            dropout=getattr(config, 'te_dropout', 0.1),
            seq_pool=getattr(config, 'seq_pool', True),
            positional_embedding=positional_embedding,
            sequence_length=getattr(config, 'sequence_length', None)
        )
        print('ECG_Transformer model initialized')
    else:
        model = models_module.ECG_Model(out_dim=config.num_classes)
        print('ECG_Model initialized')
    
    return model


def normalize_calibration_artifact(artifact):
    artifact = dict(artifact)
    if 'method' not in artifact:
        if 'temperature' in artifact:
            artifact['method'] = 'temperature'
            artifact['parameters'] = {'temperature': artifact['temperature']}
        else:
            raise ValueError("Calibration artifact must include 'method' or 'temperature'")
    if 'parameters' not in artifact:
        raise ValueError("Calibration artifact is missing 'parameters'")
    return artifact


def load_calibration(path):
    with open(path, 'r') as f:
        return normalize_calibration_artifact(json.load(f))


def find_calibration_near_model(model_path):
    model_dir = os.path.dirname(os.path.abspath(model_path))
    for name in ('calibration.json', 'temperature.json'):
        candidate = os.path.join(model_dir, name)
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_calibration_path(model_path, uncalibrated=False, calibration_path=None):
    if uncalibrated:
        return None
    if calibration_path is not None:
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        return calibration_path
    found = find_calibration_near_model(model_path)
    if found is None:
        raise FileNotFoundError(
            "Calibrated inference is the default but no calibration.json or temperature.json "
            f"was found next to the model: {os.path.dirname(os.path.abspath(model_path))}. "
            "Use --uncalibrated to return raw sigmoid(logit) probabilities."
        )
    return found


def apply_calibration(logits, artifact):
    artifact = normalize_calibration_artifact(artifact)
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits, dtype=np.float64)
    logits_np = logits_np.reshape(-1)

    method = artifact['method']
    params = artifact['parameters']

    if method == 'temperature':
        scaled = logits_np / float(params['temperature'])
        return torch.sigmoid(torch.tensor(scaled)).numpy()

    if method == 'platt':
        scaled = float(params['weight']) * logits_np + float(params['bias'])
        return torch.sigmoid(torch.tensor(scaled)).numpy()

    if method == 'isotonic':
        uncal = torch.sigmoid(torch.tensor(logits_np)).numpy()
        return np.interp(
            uncal,
            np.asarray(params['x_thresholds'], dtype=np.float64),
            np.asarray(params['y_thresholds'], dtype=np.float64),
        )

    raise ValueError(f"Unknown calibration method: {method}")


def save_model(current_epoch, model, optimizer, criterion, config_dict=None):
    os.makedirs(config.output_dir, exist_ok=True)
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M")
    
    # Save model checkpoint
    torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{config.output_dir}/model_{datetime_str}_epoch_{current_epoch}.pth')
    
    # Save config dictionary if provided
    if config_dict is not None:
        config_filename = f'{config.output_dir}/config_{datetime_str}.json'
        with open(config_filename, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)
    
    
    
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    os.makedirs(config.output_dir, exist_ok=True)
    now = datetime.datetime.now()
    # accuracy plots
    plt.figure(figsize=(10, 10))
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
    plt.savefig(f'{config.output_dir}/accuracy_{now.strftime("%Y-%m-%d_%H-%M")}.png')
    
    # loss plots
    plt.figure(figsize=(10, 10))
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
    plt.savefig(f'{config.output_dir}/loss_{now.strftime("%Y-%m-%d_%H-%M")}.png')
    
    
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
