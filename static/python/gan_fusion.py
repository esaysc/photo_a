import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 编码器部分
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 解码器部分
        self.deconv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        # 残差连接
        self.skip1 = nn.Conv2d(64, 64, kernel_size=1)
        self.skip2 = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, x):
        # 编码
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # 解码（带残差连接）
        x = F.relu(self.deconv1(x3))
        x = x + self.skip2(x2)
        x = F.relu(self.deconv2(x))
        x = x + self.skip1(x1)
        x = self.deconv3(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

def gan_fusion(visible_img, infrared_img):
    # 转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # 数据预处理
    visible_img = visible_img.astype(np.float32) / 255.0
    infrared_img = infrared_img.astype(np.float32) / 255.0
    
    # 转换为PyTorch张量
    visible_tensor = torch.from_numpy(visible_img).unsqueeze(0)
    infrared_tensor = torch.from_numpy(infrared_img).unsqueeze(0)
    
    # 拼接输入
    input_tensor = torch.cat([visible_tensor, infrared_tensor], dim=0).unsqueeze(0)
    
    # 创建生成器
    generator = Generator()
    generator.eval()
    
    # 生成融合图像
    with torch.no_grad():
        fused_image = generator(input_tensor)
        fused_image = fused_image.squeeze().numpy()
    
    # 后处理
    fused_image = (fused_image + 1) * 127.5  # 从[-1,1]转换到[0,255]
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    
    return fused_image