import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, kernel_size=(7,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=(2,1), padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                             stride=1, padding='same', bias=False)
        self.mp = nn.MaxPool2d(kernel_size=(2,1), padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = self.mp(self.skip(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)) + identity)
        return x

class TempAxis(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = [
            Block(1, 16, padding=(3,0)),
            Block(16, 16, padding=(3,0)),
            Block(16, 16, padding=(3,0)),
            nn.Dropout(0.2),
            Block(16, 32, padding=(2,0), kernel_size=(5,1)),
            Block(32, 32, padding=(2,0), kernel_size=(5,1)),
            Block(32, 32, padding=(2,0), kernel_size=(5,1)),
            nn.Dropout(0.2),
            Block(32, 64, padding=(1,0), kernel_size=(3,1)),
            Block(64, 64, padding=(1,0), kernel_size=(3,1)),
            Block(64, 64, padding=(1,0), kernel_size=(3,1)),
            nn.Dropout(0.2)
        ]
        self.layer = nn.Sequential(*blocks)
        self.pad = nn.ZeroPad2d((0,0,60,60))

    def sequence_length(self, n_channels=1, height=5000, width=8):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
    
    def forward(self, x):
        return self.layer(self.pad(x))

class LeadAxis(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.lead_axis = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,8)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1280, out_dim)
        )
        
    def forward(self, x):
        return self.lead_axis(x)

class ECG_Model(nn.Module):   
    def __init__(self, out_dim=1):
        super().__init__()
        self.tokenizer = TempAxis()
        self.lead_axis = LeadAxis(out_dim)
        
    def forward(self, x):
        return self.lead_axis(self.tokenizer(x.unsqueeze(1)))


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention and feed-forward."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, src):
        # Self-attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # Feed-forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class ECG_Transformer(nn.Module):
    """ECG Transformer model using compact convolutional transformer format."""
    def __init__(self, num_classes=1, num_encoder_layers=6, d_model=64, nhead=8, 
                 dim_feedforward=2048, dropout=0.1, seq_pool=True, 
                 positional_embedding='sine', sequence_length=None):
        super().__init__()
        
        # Validate positional_embedding parameter
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        
        # Tokenizer: first 6 blocks of TempAxis + ZeroPad2d
        tokenizer_blocks = [
            Block(1, 16, padding=(3,0)),
            Block(16, 16, padding=(3,0)),
            Block(16, 16, padding=(3,0)),
            nn.Dropout(0.2),
            Block(16, 32, padding=(2,0), kernel_size=(5,1)),
            Block(32, 32, padding=(2,0), kernel_size=(5,1)),
            Block(32, 32, padding=(2,0), kernel_size=(5,1))
        ]
        self.tokenizer = nn.Sequential(*tokenizer_blocks)
        self.pad = nn.ZeroPad2d((0,0,60,60))
        self.lead_collapse = nn.Conv2d(32, d_model, kernel_size=(1, 8), stride=1, padding=0)
        
        self.embedding_dim = d_model
        self.seq_pool = seq_pool
        self.num_tokens = 0
        
        # Compute sequence_length if not provided
        if sequence_length is None:
            if positional_embedding != 'none':
                # Compute sequence length by running a dummy input through tokenizer
                with torch.no_grad():
                    dummy_input = torch.zeros((1, 1, 5000, 8))
                    tokenized = self.tokenizer(self.pad(dummy_input))
                    collapsed = self.lead_collapse(tokenized)
                    sequence_length = collapsed.shape[2]  # Height dimension after collapse
                self.sequence_length = sequence_length
            else:
                self.sequence_length = None
        else:
            self.sequence_length = sequence_length
        
        # Assert sequence_length is provided if positional_embedding is not 'none'
        assert self.sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."
        
        # Handle seq_pool option
        pos_emb_length = self.sequence_length
        if not seq_pool:
            if pos_emb_length is not None:
                pos_emb_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                         requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = nn.Linear(d_model, 1)
        
        # Handle positional encoding
        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, pos_emb_length, d_model),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:  # 'sine'
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(pos_emb_length, d_model),
                    requires_grad=False
                )
        else:
            self.positional_emb = None
        
        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            )
        self.transformer = nn.Sequential(*encoder_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        self.fc = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.apply(self.init_weight)
        
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        """Create sinusoidal positional embeddings."""
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
    
    @staticmethod
    def init_weight(m):
        """Initialize weights similar to TransformerClassifier."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Tokenizer: (B, 1, H, W) -> (B, C, H', W')
        # Add channel dimension if input is 3D (B, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.tokenizer(self.pad(x))
        
        # Collapse along lead axis: (B, 64, H', W') -> (B, d_model, H', 1)
        x = self.lead_collapse(x)
        
        # Reshape to sequence: (B, d_model, H', 1) -> (B, H', d_model)
        B, C, H, W = x.shape
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, seq_len, d_model)
        
        # Handle padding if positional_emb is None and sequence is shorter than expected
        if self.positional_emb is None and self.sequence_length is not None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.sequence_length - x.size(1)), mode='constant', value=0)
        
        # Add class token if seq_pool is False
        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        # Add positional encoding if enabled
        if self.positional_emb is not None:
            x = x + self.positional_emb
        
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)  # (B, seq_len, d_model)
        x = self.norm(x)
        
        # Sequence pooling: (B, seq_len, d_model) -> (B, d_model)
        if self.seq_pool:
            # Attention-based pooling
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            # Use class token (first token)
            x = x[:, 0]
        
        # Classification
        x = self.fc(x)  # (B, num_classes)
        
        return x

