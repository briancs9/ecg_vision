import torch
import datetime
import torch.nn as nn
import datetime

class Config():
    
    epochs = 30
    warmup_epochs = 10
    transformer_model = 'conv'
    
    ## data paths
    train_data_path = 'test_data_set/'
    test_data_path = 'test_data_set/'
    output_dir = 'outputs_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    criterion = nn.BCELoss()
    
    ## model parameters
    num_classes = 1
    num_transformer_layers = 3
    num_heads = 8
    d_model = 256
    
    assert d_model % num_heads == 0    
    
    seq_pool = True
    batch_size = 32
    learning_rate = 5e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mlp_ratio = 4
    te_dropout = 0.1
    

    
