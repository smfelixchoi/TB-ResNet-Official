import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlock, BottleNeck
from TBResblock import TBBasicBlock, TBBottleNeck

from utils.pooling import *

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x




class BTResNet(nn.Module):
    def __init__(self, block1, block2, num_block, attention, stats, init_weights=True):
        super(BTResNet, self).__init__()
                
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation
        
        self.in_channels=64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,2), padding=1)

        self.conv2_x = self._make_layer(block2, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block2, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block2, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block2, 512, num_block[3], 2)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(5,1), stride=1, padding=0, bias=False, groups=512)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.pool = StatisticsPooling(512, attention=attention, stats=stats)
    
        if 'mu' in stats and 'std' in stats:
            self.d = 512*2
        else:
            self.d = 512
            
        self.bn3 = nn.BatchNorm1d(self.d)
        self.fc = nn.Linear(self.d, 192)
        self.bn4 = nn.BatchNorm1d(192)

        # weights initialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)
                
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = torch.squeeze(x, dim=2)
        x = self.pool(x)
        
        x = self.bn3(x)
        x = self.fc(x)
        x = self.bn4(x)
        
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18(attention, stats):
    return BTResNet(BasicBlock, TBBasicBlock, [2,2,2,2], attention, stats)

def resnet34(attention, stats):
    return BTResNet(BasicBlock, TBBasicBlock, [3, 4, 6, 3], attention, stats)

def resnet50(attention, stats):
    return BTResNet(BottleNeck, TBBottleNeck, [3, 4, 6, 3], attention, stats)

def resnet101(attention, stats):
    return BTResNet(BottleNeck, TBBottleNeck, [3, 4, 23, 3], attention, stats)

def resnet152(attention, stats):
    return BTResNet(BottleNeck, TBBottleNeck, [3, 8, 36, 3], attention, stats)