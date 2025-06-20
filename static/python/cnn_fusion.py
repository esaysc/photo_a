import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class CNNFusion(nn.Module):
    def __init__(self):
        super(CNNFusion, self).__init__()
        # 特征提取层（平衡通道数与计算量）
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        # 融合层（带注意力机制）
        self.conv_fusion = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        # 重建层（自然色彩映射）
        self.conv_recon = nn.Conv2d(96, 3, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # 更自然的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  #  Xavier初始化提升特征多样性
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 移除固定偏置避免过亮

    def forward(self, x1, x2):
        # 输入色彩柔和增强（避免过度饱和）
        x1 = torch.pow(x1, 0.9)  # 轻微伽马校正
        x2 = torch.pow(x2, 0.9)
        
        # 特征提取（加入批归一化稳定训练）
        feat1 = F.relu(self.conv1(x1))
        feat1 = F.relu(self.conv2(feat1))
        feat2 = F.relu(self.conv1(x2))
        feat2 = F.relu(self.conv2(feat2))
        
        # 注意力引导融合（抑制灰度特征对色彩的干扰）
        fused_feat = torch.cat([feat1, feat2], dim=1)
        attention = torch.sigmoid(self.conv_fusion(fused_feat))
        fused_feat = feat1 * attention + feat2 * (1 - attention)
        
        # 自然重建（降低残差权重）
        out = self.conv_recon(fused_feat)
        out = out + (x1 * 0.6 + x2 * 0.4)  # 可见光占60%，红外占40%
        out = torch.clamp(out, 0, 1)
        return out

def cnn_fusion(visible_img, infrared_img):
    # ------------------- 输入预处理（温和色彩调整） -------------------
    # 可见光：适度提升饱和度与亮度
    if len(visible_img.shape) == 2:
        visible_color = cv2.cvtColor(visible_img, cv2.COLOR_GRAY2BGR)
    else:
        visible_color = visible_img[:, :, :3].copy()
    
    hsv_vis = cv2.cvtColor(visible_color, cv2.COLOR_BGR2HSV)
    hsv_vis[..., 1] = np.clip(hsv_vis[..., 1] * 1.3, 0, 255)  # 饱和度提升30%（较之前降低）
    hsv_vis[..., 2] = cv2.equalizeHist(hsv_vis[..., 2])  # 仅均衡化，不提升亮度
    visible_color = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR)
    
    # 红外：保留低饱和伪彩色（增强色彩兼容性）
    if len(infrared_img.shape) == 2:
        infrared_color = cv2.applyColorMap(infrared_img, cv2.COLOR_MAP_BONE)  # 更柔和的色卡
        hsv_ir = cv2.cvtColor(infrared_color, cv2.COLOR_BGR2HSV)
        hsv_ir[..., 1] = np.clip(hsv_ir[..., 1] * 0.5, 0, 255)  # 饱和度保留50%
        infrared_color = cv2.cvtColor(hsv_ir, cv2.COLOR_HSV2BGR)
    else:
        infrared_color = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2RGB)[:, :, :3]
    
    # ------------------- 归一化与张量转换 -------------------
    visible_tensor = torch.from_numpy(visible_color.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    infrared_tensor = torch.from_numpy(infrared_color.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    
    # ------------------- 模型推理 -------------------
    model = CNNFusion()
    model.eval()
    
    with torch.no_grad():
        fused_tensor = model(visible_tensor, infrared_tensor)
    
    # ------------------- 后处理（自然色彩恢复） -------------------
    fused_image = fused_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    fused_image = np.clip(fused_image * 255, 0, 255).astype(np.uint8)
    
    # 色彩平衡（在LAB空间调整）
    lab_fused = cv2.cvtColor(fused_image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_fused)
    l = cv2.GaussianBlur(l, (5, 5), 0)  # 平滑亮度通道
    lab_fused = cv2.merge([l, a, b])
    fused_image = cv2.cvtColor(lab_fused, cv2.COLOR_Lab2BGR)
    
    return fused_image

# ------------------- 示例用法 -------------------
if __name__ == "__main__":
    visible_img = cv2.imread("visible.jpg", cv2.IMREAD_COLOR)
    infrared_img = cv2.imread("infrared.png", 0)
    
    fused_result = cnn_fusion(visible_img, infrared_img)
    
    print("Fused image shape:", fused_result.shape)
    cv2.imwrite("fused_natural_color.jpg", fused_result)
    cv2.imshow("Natural Color Fusion", fused_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()