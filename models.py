import torch
import torch.nn as nn



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
        
        self.skip = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=1,
                             padding='same',
                             bias=False)
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



class TempAxis(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(Block(1, 16, padding=(3,0)), 
                                   Block(16,16, padding=(3,0)), 
                                   Block(16,16, padding=(3,0)),
                                   nn.Dropout(0.2),
                                   Block(16,32, padding=(2,0), kernel_size=(5,1)),
                                   Block(32,32, padding=(2,0), kernel_size=(5,1)),
                                   Block(32,32, padding=(2,0), kernel_size=(5,1)),
                                   nn.Dropout(0.2),
                                   Block(32,64, padding=(1,0), kernel_size=(3,1)),
                                   Block(64,64, padding=(1,0), kernel_size=(3,1)),
                                   Block(64,64, padding=(1,0), kernel_size=(3,1)),
                                   nn.Dropout(0.2))
        self.pad = nn.ZeroPad2d((0,0,60,60))

    def sequence_length(self, n_channels=1, height=5000, width=8):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
    
    def forward(self, x):
        x = self.pad(x)
        x = self.layer(x)
        
        return x
    


class LeadAxis(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.lead_axis = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,8)),
                                       nn.Dropout(0.2),
                                       nn.Flatten(),
                                       nn.Linear(1280,320),
                                       nn.Linear(320, self.out_dim))
        
    def forward(self, x):
        x = self.lead_axis(x)
        return x
 
    
class ECG_Model(nn.Module):   
    def __init__(self, out_dim=1):
        super().__init__()
        self.tokenizer = TempAxis()
        self.lead_axis = LeadAxis(out_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #x = x.float()
        x = self.tokenizer(x)
        x = self.lead_axis(x)
        x = self.sigmoid(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, kernel_size=(7,1)):
        super().__init__()
        # Calculate output padding to ensure correct size
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=(2,1),
                                       padding=padding,
                                       output_padding=(1,0),  # Add output padding
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.ConvTranspose2d(in_channels=out_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=(0,0),
                                       bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=(2,1),
                                      padding=0,
                                      output_padding=(1,0),  # Add output padding
                                      bias=False)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add a size check and adjustment if needed
        if out.size() != identity.size():
            # Adjust the larger tensor to match the smaller one
            diff = out.size(2) - identity.size(2)
            if diff > 0:
                out = out[:, :, :-diff, :]
            else:
                identity = identity[:, :, :diff, :]
                
        out = self.relu(out + identity)
        return out  



class TempAxisDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Mirror the encoder layers in reverse order
        self.layer = nn.Sequential(
            nn.Dropout(0.2),
            DecoderBlock(64,32, padding=(1,0), kernel_size=(3,1)),
            DecoderBlock(32,32, padding=(1,0), kernel_size=(3,1)),
            DecoderBlock(32,32, padding=(1,0), kernel_size=(3,1)),
            nn.Dropout(0.2),
            DecoderBlock(32,16, padding=(2,0), kernel_size=(5,1)),
            DecoderBlock(16,16, padding=(2,0), kernel_size=(5,1)),
            DecoderBlock(16,16, padding=(2,0), kernel_size=(5,1)),
            nn.Dropout(0.2),
            DecoderBlock(16,1, padding=(3,0)),
            DecoderBlock(1,1, padding=(3,0)),
            DecoderBlock(1,1, padding=(3,0))
        )
        
    def forward(self, x):
        x = self.layer(x)
        # Remove the padding that was added in the encoder
        x = x[:, :, 60:-60, :]
        return x

class LeadAxisDecoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lead_axis = nn.Sequential(
            # Reverse the linear layers first
            nn.Linear(in_dim, 320),
            nn.ReLU(),
            nn.Linear(320, 1280),
            nn.ReLU(),
        )
        
        # Separate the conv operations for better control
        self.unflatten = nn.Unflatten(1, (128, 10))
        self.conv_adjust = nn.Conv2d(128, 128, kernel_size=1)
        self.conv_transpose = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1,8))
        
    def forward(self, x):
        # If input is not already flattened, flatten it
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # Apply linear layers
        x = self.lead_axis(x)
        
        # Reshape and apply convolutions
        x = self.unflatten(x)
        x = x.unsqueeze(-1)  # Add an extra dimension for the conv2d
        x = self.conv_adjust(x)
        x = self.conv_transpose(x)
        
        return x



class ECG_Autoencoder(nn.Module):   
    def __init__(self):
        super().__init__()
        # Encoders
        self.temp_encoder = TempAxis()
        self.lead_encoder = LeadAxis(out_dim=1200)  # Adjust out_dim as needed
        
        # Decoders
        self.lead_decoder = LeadAxisDecoder(in_dim=1200)  # Match out_dim from encoder
        self.temp_decoder = TempAxisDecoder()
        
    def forward(self, x):
        # Encoding
        temp_encoded = self.temp_encoder(x)
        lead_encoded = self.lead_encoder(temp_encoded)
        
        # Decoding
        lead_decoded = self.lead_decoder(lead_encoded)
        temp_decoded = self.temp_decoder(lead_decoded)
        
        return temp_decoded, lead_encoded  # Return both the reconstruction and encoded representation
    

    

