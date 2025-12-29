import torch
import datetime
import torch.nn as nn
import json
import os

class Config():
    def __init__(self, config_json_path=None):
        ##training run parameters
        self.epochs = 30
        self.warmup = 3
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.annealing = 1e-2
        self.transformer_model = 'conv'  # options: 'transformer' or 'conv'
        self.train_annotations_file = 'annotations/train_annotations.csv'
        self.val_annotations_file = 'annotations/val_annotations.csv'
        self.pos_weight = 16.
        self.num_classes = 1
        self.d_model = 256
        self.num_heads = 8
        self.te_dropout = 0.1
        self.use_pos_encoding = True
        self.num_transformer_layers = 6
        self.weight_decay = 1e-4
        self.num_classes = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        ## data paths
        self.output_dir = 'outputs_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.data_path = '/data1/shaffeb1/csv_data/'
        
        # Load parameters from JSON file if provided
        if config_json_path is not None:
            self.load_from_json(config_json_path)
    
    def load_from_json(self, json_path):
        """
        Load configuration parameters from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Config JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Update attributes from loaded JSON
        for key, value in loaded_config.items():
            if hasattr(self, key):
                # Special handling for certain attributes
                if key == 'device':
                    # Device should be determined at runtime, but allow override
                    if value in ['cuda', 'cpu', 'mps']:
                        if value == 'cuda' and not torch.cuda.is_available():
                            print(f"Warning: CUDA requested but not available. Using CPU instead.")
                            value = 'cpu'
                        elif value == 'mps' and not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                            print(f"Warning: MPS requested but not available. Using CPU instead.")
                            value = 'cpu'
                        self.device = value
                    else:
                        # Fallback to default device detection
                        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                elif key == 'output_dir':
                    # Don't override output_dir with old timestamp, keep new one
                    # Or optionally use the loaded one if you want to resume to same directory
                    # self.output_dir = value  # Uncomment if you want to use the loaded output_dir
                    pass
                else:
                    # For other attributes, directly assign the value
                    setattr(self, key, value)
            else:
                # If key doesn't exist in current config, add it (for forward compatibility)
                setattr(self, key, value)
        
        # Re-validate transformer-specific parameters if transformer model is selected
        if self.transformer_model == 'transformer':
            if self.d_model % self.num_heads != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        
        print(f"Configuration loaded from: {json_path}")



