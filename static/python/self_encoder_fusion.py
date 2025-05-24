import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

class DenseFuse(nn.Module):
    def __init__(self):
        super(DenseFuse, self).__init__()
        # 编码器
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 解码器
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
        
    def decoder(self, x):
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
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
    
    # 创建模型并加载预训练权重
    model = DenseFuse()
    weights_path = os.path.join(os.path.dirname(__file__), '../model/model_weight.pkl')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # 特征提取和融合
    with torch.no_grad():
        visible_features = model.encoder(visible_tensor)
        infrared_features = model.encoder(infrared_tensor)
        
        # 自适应权重融合
        fusion_weight = torch.sigmoid(visible_features + infrared_features)
        fused_features = fusion_weight * visible_features + (1 - fusion_weight) * infrared_features
        
        # 解码融合特征
        fused_image = model.decoder(fused_features)
        fused_image = torch.tanh(fused_image) * 0.5 + 0.5
    
    # 后处理
    fused_image = fused_image.squeeze().numpy()
    fused_image = np.clip(fused_image * 255, 0, 255).astype(np.uint8)
    
    return fused_image

# 示例使用
if __name__ == "__main__":
    visible_img = cv2.imread('visible_image.jpg')  # 替换为可见光图像的路径
    infrared_img = cv2.imread('infrared_image.jpg')  # 替换为红外图像的路径
    
    fused_image = self_encoder_fusion(visible_img, infrared_img)
    
    # 显示或保存融合图像
    cv2.imshow('Fused Image', fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()