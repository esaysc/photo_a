import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def self_encoder_fusion(visible_img, infrared_img):
    # 转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # 数据预处理
    visible_img = visible_img.astype(np.float32) / 255.0
    infrared_img = infrared_img.astype(np.float32) / 255.0
    
    # 转换为PyTorch张量
    visible_tensor = torch.from_numpy(visible_img).unsqueeze(0).unsqueeze(0)
    infrared_tensor = torch.from_numpy(infrared_img).unsqueeze(0).unsqueeze(0)
    
    # 创建并初始化自编码器
    model = AutoEncoder()
    
    # 编码
    with torch.no_grad():
        visible_features = model.encoder(visible_tensor)
        infrared_features = model.encoder(infrared_tensor)
        
        # 特征融合（加权平均）
        fused_features = visible_features * 0.5 + infrared_features * 0.5
        
        # 解码
        fused_image = model.decoder(fused_features)
    
    # 后处理
    fused_image = fused_image.squeeze().numpy()
    fused_image = np.clip(fused_image * 255, 0, 255).astype(np.uint8)
    
    return fused_image