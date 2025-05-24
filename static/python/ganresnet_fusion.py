import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 编码器
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # ResNet块
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.res3 = ResBlock(128)
        
        # 解码器
        self.deconv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 编码
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # ResNet特征提取
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # 解码
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

def ganresnet_fusion(visible_img, infrared_img):
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
    
    # 创建生成器模型并加载预训练权重
    model = Generator()
    weights_path = os.path.join(os.path.dirname(__file__), 'ganresnet_weights.pkl')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # 特征提取和融合
    with torch.no_grad():
        # 提取特征
        visible_features = model(visible_tensor)
        infrared_features = model(infrared_tensor)
        
        # 自适应权重融合
        attention = torch.sigmoid(visible_features * infrared_features)
        fused_features = attention * visible_features + (1 - attention) * infrared_features
        
        # 生成融合图像
        fused_image = torch.tanh(fused_features) * 0.5 + 0.5
    
    # 后处理
    fused_image = fused_image.squeeze().numpy()
    fused_image = np.clip(fused_image * 255, 0, 255).astype(np.uint8)
    
    return fused_image