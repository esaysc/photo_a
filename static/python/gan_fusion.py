import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ColorPreserveBlock(nn.Module):
    def __init__(self, channels=64):
        super(ColorPreserveBlock, self).__init__()
        self.vis_conv = nn.Conv2d(3, channels, 1)  # 可见光色彩特征
        self.ir_conv = nn.Conv2d(3, channels, 1)   # 红外强度特征
        self.fusion = nn.Conv2d(channels*2, 3, 1)  # 色彩融合
    
    def forward(self, vis, ir):
        vis_feat = F.relu(self.vis_conv(vis))
        ir_feat = F.relu(self.ir_conv(ir))
        concat = torch.cat([vis_feat, ir_feat], dim=1)
        color_map = torch.sigmoid(self.fusion(concat))  # 生成色彩映射
        return color_map * vis + (1 - color_map) * ir   # 自适应融合

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        return out + residual  # 带BN的残差连接

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 可见光通道特征提取
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            EnhancedResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            EnhancedResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            EnhancedResidualBlock(256)
        )
        
        # 红外通道特征提取
        self.ir_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            EnhancedResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            EnhancedResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            EnhancedResidualBlock(256)
        )
        
        # 特征融合与解码
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            EnhancedResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            EnhancedResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            EnhancedResidualBlock(64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # 色彩保留模块
        self.color_preserve = ColorPreserveBlock()

    def forward(self, vis, ir):
        # 分别提取特征
        vis_feat = self.vis_encoder(vis)
        ir_feat = self.ir_encoder(ir)
        
        # 特征融合
        concat_feat = torch.cat([vis_feat, ir_feat], dim=1)
        
        # 解码生成融合图像
        fused = self.decoder(concat_feat)
        
        # 色彩保留增强
        color_enhanced = self.color_preserve(vis, fused)
        return color_enhanced

def gan_fusion(visible_img, infrared_img):
    # 强制转为彩色并增强对比度
    def enhance_color(img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 自适应色彩增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 亮度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # 色彩增强（通过HSV空间）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 15)  # 饱和度增强
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv_enhanced = cv2.merge((h, s, v))
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 转回LAB合并增强后的亮度
        lab_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2LAB)
        l_enhanced, a_enhanced, b_enhanced = cv2.split(lab_enhanced)
        final = cv2.merge((cl, a_enhanced, b_enhanced))
        
        return cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    
    # 预处理增强
    visible_enhanced = enhance_color(visible_img)
    infrared_enhanced = enhance_color(infrared_img)
    
    # 尺寸统一
    h, w = visible_enhanced.shape[:2]
    infrared_enhanced = cv2.resize(infrared_enhanced, (w, h))
    
    # 归一化到[-1, 1]
    visible_normalized = (visible_enhanced.astype(np.float32) / 127.5) - 1.0
    infrared_normalized = (infrared_enhanced.astype(np.float32) / 127.5) - 1.0
    
    # 转换为张量
    vis_tensor = torch.from_numpy(visible_normalized).permute(2, 0, 1).unsqueeze(0)
    ir_tensor = torch.from_numpy(infrared_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # 生成器推理
    generator = Generator()
    generator.eval()
    
    with torch.no_grad():
        fused_tensor = generator(vis_tensor, ir_tensor)
        fused_image = fused_tensor.squeeze().cpu().numpy()
    
    # 后处理
    fused_image = (fused_image + 1.0) * 127.5
    fused_image = np.transpose(fused_image, (1, 2, 0)).astype(np.uint8)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_RGB2BGR)
    
    # 清晰度增强
    blurred = cv2.GaussianBlur(fused_image, (5,5), 0)
    sharpened = cv2.addWeighted(fused_image, 1.7, blurred, -0.7, 0)
    
    # 最终色彩平衡
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, 2)  # 微调红色-绿色轴
    b = cv2.add(b, 2)  # 微调蓝色-黄色轴
    final = cv2.merge((l, a, b))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    
    return final

# 示例使用
if __name__ == "__main__":
    visible_img = cv2.imread("visible.jpg")
    infrared_img = cv2.imread("infrared.jpg", 0)
    
    fused_image = gan_fusion(visible_img, infrared_img)
    cv2.imshow("Colorful and Sharp Fused Image", fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()