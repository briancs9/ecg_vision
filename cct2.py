import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, kernel_size=(7,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=1,
                              kernel_size=kernel_size,
                              padding='same',
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=(2,1),
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=1,
                             padding='same',
                             bias=False),
                                  nn.BatchNorm2d(out_channels))
        self.mp = nn.MaxPool2d(kernel_size=(2,1), padding=0)
        
    def forward(self, x):
        identity = x
        identity = self.skip(identity)
        identity = self.mp(identity)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)
        
        return x


class Embedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(Block(in_channels=1, out_channels=16, padding=(3,0)),
                                       Block(in_channels=16, out_channels=32, padding=(3,0)),
                                       Block(in_channels=32, out_channels=64, padding=(3,0)),
                                       nn.MaxPool2d(kernel_size=(2,1), padding=(0,0)))
        
        self.lead_axis = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1,12))
        self.flatten = nn.Flatten(2,3)
        self.apply(self.init_weight)
    
    def sequence_length(self):
        return self.forward(torch.zeros((1, 1, 5000, 12))).shape[1]
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.lead_axis(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1) #(batch_size, sequence_length, d_model)
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=num_heads,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, x):
        x = self.pre_norm(x)
        x = x + self.drop_path(self.self_attn(x))
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + self.drop_path(x)
        return x




class CCT(nn.Module):
    def __init__(self,
                 num_classes=1, 
                 num_heads=8, 
                 num_transformer_layers=3, 
                 d_model=256,
                 te_dropout=0.1,
                 seq_pool=True):
        
        super(CCT, self).__init__()
        
        self.embedding = Embedding()
        self.sequence_length = self.embedding.sequence_length()
        self.num_classes = num_classes
        self.seq_pool = seq_pool
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.positional_emb = nn.Parameter(self.sinusoidal_embedding(self.sequence_length, d_model), requires_grad=False)
        
        self.blocks = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, 
                             num_heads=num_heads, 
                             dropout=te_dropout) for _ in range(num_transformer_layers)])
        self.mlp_head = nn.Linear(d_model, num_classes)
        
        self.attention_pool = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x += self.positional_emb
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = torch.mean(x, dim=1)
        
        x = self.mlp_head(x)
        x = torch.sigmoid(x)
        return x
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

