import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CCT(nn.Module):
    def __init__(self, 
                 num_classes=1, 
                 num_heads=12, 
                 num_transformer_layers=7, 
                 d_model=128,
                 mlp_ratio=4,
                 te_dropout=0.1,
                 seq_pool=True):
        
        super(CCT, self).__init__()
        dim_feedforward = int(d_model * mlp_ratio)
        
        self.embedding = nn.Sequential(
                ResidualBlock(12, 32, kernel_size=7),
                nn.MaxPool1d(kernel_size=2, stride=2),
                ResidualBlock(32, 64, kernel_size=5),
                nn.MaxPool1d(kernel_size=2, stride=2),
                ResidualBlock(64, d_model, kernel_size=3),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
        self.num_classes = num_classes
        self.seq_pool = seq_pool
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn_output_size = d_model * (5000 // 8)  # After 3 pooling layers: 5000 // 2^3
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=te_dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_transformer_layers)
        
        self.attention_pool = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        x = self.transformer_encoder(x)
        
        x = self.norm(x)
        
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = torch.mean(x, dim=1)
        
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
