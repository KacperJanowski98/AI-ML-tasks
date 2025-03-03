import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDoubleConv(nn.Module):
    """Double convolution block with residual connection for U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # Main path with double convolution
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection (1x1 conv if dimensions don't match)
        self.residual_connection = nn.Sequential()
        if in_channels != out_channels:
            self.residual_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.double_conv(x)
        return F.relu(out + residual)  # Add residual connection


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First encoder block doesn't follow a pooling
        self.encoder.append(ResidualDoubleConv(in_channels, features[0]))
        
        # Rest of encoder blocks
        for i in range(1, len(features)):
            self.encoder.append(ResidualDoubleConv(features[i-1], features[i]))
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        for i in range(len(features)-1, 0, -1):
            self.upconv.append(
                nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2)
            )
            self.decoder.append(
                ResidualDoubleConv(features[i], features[i-1])
            )
        
        # Final 1x1 convolution to produce segmentation map
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Remove the last skip connection (the one before bottleneck)
        skip_connections = skip_connections[:-1]
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder):
            x = self.upconv[i](x)
            skip = skip_connections[i]
            
            # Handle case when dimensions don't match exactly
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            # Concatenate skip connection with upsampled features
            x = torch.cat((skip, x), dim=1)  # Skip connection
            x = decoder_block(x)
        
        # Final 1x1 convolution
        return self.final_conv(x)
