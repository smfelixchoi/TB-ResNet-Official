import numpy as np

import torch
from torch import nn

class StatisticsPooling(nn.Module):
    def __init__(self, f, attention='channel', stats='mu_std'):
        '''
        Possible arguments for attention: 'channel', 'temporal', False
        Channel attentive pooling: Proposed in ECAPA-TDNN, give different attention weights to each channel (in frequency axis)
        Temporal attentive pooling: Give the same attention weights to every channel.
        False: No attention. 
        '''
        
        super(StatisticsPooling, self).__init__()
        self.f = f
        self.d = int(f/8)
        self.attention = attention
        self.stats=stats
        
        if attention == 'channel':
            self.attention_weights = nn.Sequential(
                nn.Conv1d(self.f, self.d, kernel_size=1), 
                nn.ReLU(),
                nn.Conv1d(self.d, self.f, kernel_size=1),
                nn.Softmax(dim=2)
            )            
        elif attention == 'temporal':
            self.attention_weights = nn.Sequential(
                nn.Conv1d(self.f, self.d, kernel_size=1), 
                nn.ReLU(),
                nn.Conv1d(self.d, 1, kernel_size=1),
                nn.Softmax(dim=2)
            )            
                
    def forward(self, x):
        '''
        input shape: (B, f, t)
        '''
        if self.f != x.size(1):
            raise ValueError
            
        if self.attention == 'channel' or self.attention == 'temporal':
            w = self.attention_weights(x)
        else:
            w = 1/x.size(2)
        
        mu = torch.sum(x*w, dim=2)
        std = torch.sqrt( (torch.sum( (x**2)*w, dim=2) - mu**2).clamp(min=1e-4) )
        
        if 'mu' in self.stats and 'std' in self.stats:
            return torch.cat((mu,std),1)
        
        elif 'mu' in self.stats and 'std' not in self.stats:
            return mu
        
        elif 'mu' not in self.stats and 'std' in self.stats:
            return std
        
        else:
            raise ValueError

            
class Post_WeightedStatisticsPooling(nn.Module):
    def __init__(self, f, e, attention='channel', ratio=40):
        '''
        Possible arguments for attention: 'channel', False
        Channel attentive pooling: Proposed in ECAPA-TDNN, give different attention weights to each channel (in frequency axis)
        Temporal attentive pooling: Give the same attention weights to every channel.
        RATIO: 
        '''
        
        super(NEWStatisticsPooling, self).__init__()
        self.f = f
        self.d = int(f/8)
        self.e = e
        self.attention = attention
        self.ratio = ratio/100 
        
        self.f1 = int(self.ratio * self.e)
        self.f2 = int(self.e - self.f1)
        
        if attention == 'channel':
            self.attention_weights = nn.Sequential(
                nn.Conv1d(self.f, self.d, kernel_size=1), 
                nn.ReLU(),
                nn.Conv1d(self.d, self.f, kernel_size=1),
                nn.Softmax(dim=2)
            )
        elif attention == 'temporal':
            self.attention_weights = nn.Sequential(
                nn.Conv1d(self.f, self.d, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.d, 1, kernel_size=1),
                nn.Softmax(dim=2)
            )
       
        self.mu_weight  = nn.Linear(self.f, self.f1)
        self.std_weight = nn.Linear(self.f, self.f2)
                                       
    def forward(self, x):
        '''
        input shape: (B, f, t)
        '''
        if self.f != x.size(1):
            raise ValueError
            
        if self.attention == 'channel' or self.attention == 'temporal':
            w = self.attention_weights(x)
        else:
            w = 1/x.size(2)
        
        mu = torch.sum(x*w, dim=2)
        std = torch.sqrt( (torch.sum( (x**2)*w, dim=2) - mu**2).clamp(min=1e-4) )
        
        mu = self.mu_weight(mu)
        std = self.std_weight(std)
        
        return torch.cat((mu,std), 1)
        

class LpNormPool(nn.Module):
    def __init__(self, p, f, attention):
        super(LpNormPool, self).__init__()
        self.p = p
        self.f = f
        self.d = f//8
        self.attention=attention
        
        if attention:
            self.attention_weights = nn.Sequential(
                nn.Conv1d(self.f, self.d, kernel_size=1), 
                nn.ReLU(),
                nn.Conv1d(self.d, self.f, kernel_size=1),
                nn.Softmax(dim=2)
            )
            
    def forward(self, x):
        if self.f != x.size(1):
            raise ValueError
            
        if self.attention:
            w = self.attention_weights(x)
        else:
            w = 1/x.size(2)
        
        if self.p == 1:
            x = torch.sum(w*torch.abs(x), dim=2, keepdim=False).clamp(min=1e-4)
            return x
        
        elif self.p < 10:    
            x = torch.sum( w*torch.pow(x, self.p), dim=2, keepdim=False).clamp(min=1e-4)
            return torch.pow(x, 1/self.p)
        
        else:
            t = x.size(2)
            m = nn.MaxPool1d(kernel_size=t, stride=1)
            return m(w*x).squeeze(dim=2)
            

class MaxNormPool(nn.Module):
    def __init__(self):
        super(MaxNormPool, self).__init__()
        
    def forward(self,x):
        t = x.size(2)
        m = nn.MaxPool1d(kernel_size=t, stride=1)
        return m(x)
        