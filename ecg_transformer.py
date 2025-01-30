import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
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

class CNNTransformerHybrid(nn.Module):
    def __init__(self, num_classes=10, num_heads=8, num_transformer_layers=3):
        super(CNNTransformerHybrid, self).__init__()
        
        # CNN layers with residual blocks and decreasing kernel sizes
        self.conv1 = ResidualBlock(12, 32, kernel_size=7)
        self.conv2 = ResidualBlock(32, 64, kernel_size=5)
        self.conv3 = ResidualBlock(64, 128, kernel_size=3)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after CNN layers
        self.cnn_output_size = 128 * (5000 // 8)  # After 3 pooling layers: 5000 // 2^3
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Final fully connected layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 8, 5000)
        
        # CNN layers with residual blocks
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        
        # Prepare for Transformer (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        # Transformer layers
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x



        
        


