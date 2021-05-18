# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:34:05 2021

@author: zqq
"""

import torch
import torch.nn as nn
from torchsummary import summary

# 标准卷积
def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(# Conv + BN + ReLU
                         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.ReLU(inplace=True),
                        )

# 深度可分离卷积
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(# part1 Conv dw: depthwise convolution, 深度卷积
                         nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                         nn.BatchNorm2d(inp),
                         nn.ReLU(inplace=True), # Use ReLU Activation Function
                
                         # part2 Conv pw: pointwise convolution, 1×1卷积
                         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.ReLU(inplace=True), # Use ReLU Activation Function
                        )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(conv_bn(3, 32, 2),    # 224,224,3  -> 112,112,32
                                   conv_dw(32, 64, 1),   # 112,112,32 -> 112,112,64
                                   conv_dw(64, 128, 2),  # 112,112,64 -> 56,56,128
                                   conv_dw(128, 128, 1), # 56,56,128  -> 56,56,128
                                   conv_dw(128, 256, 2), # 56,56,128  -> 28,28,256
                                   conv_dw(256, 256, 1), # 28,28,256  -> 28,28,256
                                   
                                   conv_dw(256, 512, 2), # 28,28,256  -> 14,14,512
                                   conv_dw(512, 512, 1), # 14,14,512  -> 14,14,512
                                   conv_dw(512, 512, 1), # 14,14,512  -> 14,14,512
                                   conv_dw(512, 512, 1), # 14,14,512  -> 14,14,512
                                   conv_dw(512, 512, 1), # 14,14,512  -> 14,14,512
                                   conv_dw(512, 512, 1), # 14,14,512  -> 14,14,512
                                   
                                   conv_dw(512, 1024, 2), # 14,14,512  -> 7,7,1024
                                   conv_dw(1024, 1024, 1), # 7,7,1024 -> 7,7,1024
                                  )
            
        self.avg = nn.AdaptiveAvgPool2d((1,1)) # 自适应池化, 1,1,1024
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_v1(pretrained=False, progress=True):
    model = MobileNetV1()
    if pretrained:
        print("mobilenet_v1 has no pretrained model")
    return model

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1().to(device)
    print(model)
    summary(model, input_size=(3, 224, 224))
    # 卷积计算公式： (N-K+2P)/S + 1, N图片大小，K核大小，P是padding, S是步长
    # 步长=2, 图片大小减半
    
    
    