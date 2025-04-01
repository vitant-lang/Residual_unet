import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import webbrowser
import threading
import time
import os

from torch.utils.tensorboard import SummaryWriter
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ImprovedAutoencoderWithUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(ImprovedAutoencoderWithUNet, self).__init__()
        
        # down samping =============
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  #  H/2, W/2
        )
        
        self.encoder2 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  #  H/4, W/4
        )
        
        self.encoder3 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()  # H/4, W/4
        )
        
        # Attention 
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),  # 
            nn.Sigmoid()
        )
      
        # --- UPSAMPING ---
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  #  H/2, W/2
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.adjust1 = nn.Conv2d(256, 128, kernel_size=1)  # （dec1+enc2=256→128）
        
        # --- SECONDUOSAMPING ---
        self.decoder2 = nn.Sequential(
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  #  H, W
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.adjust2 = nn.Conv2d(128, 64, kernel_size=1)  # （dec2+enc1=128→64）
        
      
        self.decoder3 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  #  [0,1]
        )
    
    def forward(self, x):
        
        enc1 = self.encoder1(x)      # [N, 64, 128, 128]
        enc2 = self.encoder2(enc1)   # [N, 128, 64, 64]
        enc3 = self.encoder3(enc2)   # [N, 256, 64, 64]
        
       
        attn = self.attention(enc3)  # [N, 1, 64, 64]
        enc3 = enc3 * attn           # [N, 256, 64, 64]
        
        
        dec1 = self.decoder1(enc3)                          # [N, 128, 128, 128]
        enc2_upsampled = F.interpolate(enc2, scale_factor=2, mode='bilinear', align_corners=False)  # [N, 128, 128, 128]
        dec1 = torch.cat([dec1, enc2_upsampled], dim=1)     # [N, 256, 128, 128]
        dec1 = self.adjust1(dec1)                          # [N, 128, 128, 128]
        
      
        dec2 = self.decoder2(dec1)                         # [N, 64, 256, 256]
        enc1_upsampled = F.interpolate(enc1, scale_factor=2, mode='bilinear', align_corners=False)  # [N, 64, 256, 256]
        dec2 = torch.cat([dec2, enc1_upsampled], dim=1)    # [N, 128, 256, 256]
        dec2 = self.adjust2(dec2)                          # [N, 64, 256, 256]
        
        out = self.decoder3(dec2)                         # [N, 1, 256, 256]
        return out
            # INPUT，OUTPUI）
model = ImprovedAutoencoderWithUNet(in_channels=2, out_channels=1)


print(model)

writer = SummaryWriter()
writer.add_graph(model, torch.randn(1, 2, 256, 256))
writer.close()
