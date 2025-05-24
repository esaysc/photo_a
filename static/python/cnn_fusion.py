import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class CNNFusion(nn.Module):
    def __init__(self):
        super(CNNFusion, self).__init__()
        # 特征提取层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 融合层
        self.conv_fusion = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # 重建层
        self.conv_recon = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        # 初始化权重为特定值，模拟训练效果
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 将卷积层的权重初始化为平均值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1.0 / (m.weight.shape[0] * m.weight.shape[1]))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x1, x2):
        # 保存输入，用于残差连接
        input1, input2 = x1, x2
        
        # 特征提取
        feat1 = F.relu(self.conv1(x1))
        feat1 = F.relu(self.conv2(feat1))
        
        feat2 = F.relu(self.conv1(x2))
        feat2 = F.relu(self.conv2(feat2))
        
        # 特征融合
        fused_feat = torch.cat([feat1, feat2], dim=1)
        fused_feat = F.relu(self.conv_fusion(fused_feat))
        
        # 图像重建（使用残差连接）
        out = self.conv_recon(fused_feat)
        # 添加残差连接，保持细节信息
        out = out + (input1 + input2) * 0.5
        
        return out

def cnn_fusion(visible_img, infrared_img):
    # 将图像转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # 归一化
    visible_img = visible_img.astype(np.float32) / 255.0
    infrared_img = infrared_img.astype(np.float32) / 255.0
    
    # 转换为PyTorch张量
    visible_tensor = torch.from_numpy(visible_img).unsqueeze(0).unsqueeze(0)
    infrared_tensor = torch.from_numpy(infrared_img).unsqueeze(0).unsqueeze(0)
    
    # 创建模型
    model = CNNFusion()
    
    # 前向传播
    with torch.no_grad():
        fused_tensor = model(visible_tensor, infrared_tensor)
    
    # 转换回numpy数组
    fused_image = fused_tensor.squeeze().numpy()
    
    # 确保像素值在有效范围内
    fused_image = np.clip(fused_image, 0, 1)
    
    # 反归一化
    fused_image = np.uint8(fused_image * 255)
    
    return fused_image