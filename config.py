import torch
import datetime
import torch.nn as nn
import datetime

class Config():
    
    epochs = 1
    
    ## data paths
    train_data_path = 'test_data_set/'
    test_data_path = 'test_data_set/'
    output_dir = 'outputs_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    criterion = nn.BCEWithLogitsLoss()
    
    ## model parameters
    num_classes = 1
    num_transformer_layers = 7
    num_heads = 4
    d_model = 128
    
    assert d_model % num_heads == 0    
    
    seq_pool = True
    batch_size = 32
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mlp_ratio = 4
    te_dropout = 0.1
    

    
