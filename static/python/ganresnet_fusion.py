import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # 注意力机制
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        
        # 应用注意力机制
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        return out + residual  # 带注意力的残差连接

class Generator(nn.Module):
    def __init__(self, num_res_blocks=9):
        super(Generator, self).__init__()
        
        # 多尺度特征提取
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            EnhancedResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            EnhancedResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            EnhancedResidualBlock(256)
        )
        
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
        self.color_preserve = nn.Sequential(
            nn.Conv2d(6, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, vis, ir):
        # 分别提取特征
        vis_feat = self.vis_encoder(vis)
        ir_feat = self.ir_encoder(ir)
        
        # 特征融合
        concat_feat = torch.cat([vis_feat, ir_feat], dim=1)
        
        # 解码生成融合图像
        fused = self.decoder(concat_feat)
        
        # 色彩保留增强
        color_map = self.color_preserve(torch.cat([vis, fused], dim=1))
        color_enhanced = color_map * vis + (1 - color_map) * fused
        return color_enhanced

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

def preprocess_images(visible_img, infrared_img):
    # 确保图像尺寸一致
    if visible_img.shape != infrared_img.shape:
        infrared_img = cv2.resize(infrared_img, (visible_img.shape[1], visible_img.shape[0]))
    
    # 转换为浮点数并归一化到[-1, 1]
    visible_norm = (visible_img.astype(np.float32) / 127.5) - 1.0
    infrared_norm = (infrared_img.astype(np.float32) / 127.5) - 1.0
    
    # 转换为PyTorch张量
    visible_tensor = torch.from_numpy(visible_norm).permute(2, 0, 1).unsqueeze(0)
    infrared_tensor = torch.from_numpy(infrared_norm).permute(2, 0, 1).unsqueeze(0)
    
    return visible_tensor, infrared_tensor

def postprocess_image(fused_tensor):
    # 转换回numpy数组并调整范围到[0, 255]
    fused_image = fused_tensor.squeeze().cpu().numpy()
    fused_image = (fused_image + 1.0) * 127.5
    fused_image = np.transpose(fused_image, (1, 2, 0)).astype(np.uint8)
    
    # 色彩增强
    hsv = cv2.cvtColor(fused_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 饱和度增强
    s = np.clip(s * 1.2, 0, 255).astype(hsv.dtype)
    
    # 亮度调整
    v = np.clip(v * 1.05, 0, 255).astype(hsv.dtype)
    
    hsv_enhanced = cv2.merge((h, s, v))
    fused_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # 对比度增强
    lab = cv2.cvtColor(fused_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    fused_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return fused_image

def adaptive_detail_enhancement(img):
    # 自适应细节增强
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算拉普拉斯算子以检测边缘
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 自适应阈值处理
    _, binary = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 细节增强
    kernel = np.ones((3,3), np.uint8)
    enhanced = cv2.addWeighted(img, 1.2, cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel), -0.2, 0)
    
    # 仅在边缘区域应用增强
    mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    final = np.where(mask > 0, enhanced, img)
    final = np.uint8(final)
    
    return final

def ganresnet_fusion(visible_img, infrared_img):
    # 检查输入图像是否为灰度图，如果是则转换为彩色
    if len(visible_img.shape) == 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_GRAY2BGR)
    if len(infrared_img.shape) == 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)
    
    # 图像预处理
    visible_tensor, infrared_tensor = preprocess_images(visible_img, infrared_img)
    
    # 加载预训练模型
    generator = Generator()
    # 实际使用时需要加载预训练权重
    # generator.load_state_dict(torch.load('generator_weights.pth'))
    generator.eval()
    
    # 模型推理
    with torch.no_grad():
        fused_tensor = generator(visible_tensor, infrared_tensor)
    
    # 后处理
    fused_image = postprocess_image(fused_tensor)
    
    # 细节增强
    fused_image = adaptive_detail_enhancement(fused_image)
    
    # 最终色彩平衡
    lab = cv2.cvtColor(fused_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 微调红色-绿色轴和蓝色-黄色轴
    a = cv2.add(a, 3)
    b = cv2.add(b, 3)
    
    final = cv2.merge((l, a, b))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    
    return final

if __name__ == "__main__":
    # 示例使用
    visible_img = cv2.imread("visible.jpg")
    infrared_img = cv2.imread("infrared.jpg")
    
    if visible_img is None or infrared_img is None:
        print("无法加载图像，请检查图像路径")
    else:
        fused_image = ganresnet_fusion(visible_img, infrared_img)
        cv2.imshow("GAN-ResNet Fused Image", fused_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    