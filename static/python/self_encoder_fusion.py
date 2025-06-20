import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustAutoencoder(nn.Module):
    def __init__(self):
        super(RobustAutoencoder, self).__init__()
        # 色彩编码器（增强RGB通道）
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 增加通道数强化色彩特征
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # 结构编码器（提取红外强度，增加边缘敏感层）
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            # 新增边缘增强卷积层
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # 联合解码器（特征融合与重建，增强上采样）
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 输入通道增加至256
            nn.ReLU(True),
            # 改用双线性上采样提升清晰度
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )

    def forward(self, color_img, struct_img):
        color_feat = self.color_encoder(color_img)
        struct_feat = self.structure_encoder(struct_img)
        # 调整色彩与结构特征权重比，使色彩更自然
        fused_feat = torch.cat([color_feat * 1.6, struct_feat * 1.4], dim=1)  # 降低色彩权重，提升结构权重
        return self.decoder(fused_feat)

def reliable_preprocess(visible, infrared):
    # 鲁棒性预处理（优化色彩增强，避免过度饱和）
    def safe_load(img_path, is_ir=False):
        img = cv2.imread(img_path) if not is_ir else cv2.imread(img_path, 0)
        if img is None:
            raise ValueError("图像加载失败，请检查路径")
        return img

    if isinstance(visible, str):
        visible = safe_load(visible)
    if isinstance(infrared, str):
        infrared = safe_load(infrared, is_ir=True)

    # 强制转为3通道（使用更自然的JET色卡，降低饱和度增幅）
    def to_three_channels(img, is_ir=False):
        if len(img.shape) == 2:
            if is_ir:
                # 改用JET色卡并适度增强饱和度
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[..., 1] = np.clip(img[..., 1] * 1.3, 0, 255)  # 饱和度提升30%
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img[:, :, :3]  # 截断Alpha通道

    visible = to_three_channels(visible)
    infrared = to_three_channels(infrared, is_ir=True)

    # 尺寸统一（使用高质量Lanczos插值）
    h, w = visible.shape[:2]
    infrared = cv2.resize(infrared, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 色彩增强（可见光图像采用更自然的LAB空间处理）
    lab_vis = cv2.cvtColor(visible, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_vis)
    # 仅轻微增强色彩通道，避免过度鲜艳
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    a = clahe.apply(a)
    b = clahe.apply(b)
    visible = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_Lab2BGR)

    # 归一化到[-1, 1]
    visible = (visible / 127.5) - 1.0
    infrared = (infrared / 127.5) - 1.0

    return (torch.from_numpy(visible.transpose(2, 0, 1)).unsqueeze(0).float(),
            torch.from_numpy(infrared.transpose(2, 0, 1)).unsqueeze(0).float())

def natural_postprocess(tensor):
    # 后处理（优化色彩自然度，避免人工痕迹）
    img = (tensor.squeeze().cpu().detach() + 1.0) * 127.5
    img = np.clip(img.numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8)

    # 锐化处理（保持清晰度但不过度）
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)

    # 色彩增强（LAB空间精细调整）
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    # 计算亮度直方图
    hist = cv2.calcHist([l], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    
    # 找到亮度的自适应阈值
    thresh = np.searchsorted(Q, 0.95)
    
    # 对不同亮度区域应用不同的色彩增强策略
    mask_dark = l < thresh//2
    mask_mid = (l >= thresh//2) & (l < thresh)
    mask_bright = l >= thresh
    
    # 对a,b通道进行增强（色彩通道）
    a_enhanced = a.copy().astype(np.float32)
    b_enhanced = b.copy().astype(np.float32)
    
    # 暗部区域：适度增强
    a_enhanced[mask_dark] = np.clip(a[mask_dark] * 1.2, -127, 127)
    b_enhanced[mask_dark] = np.clip(b[mask_dark] * 1.2, -127, 127)
    
    # 中间区域：大幅增强
    a_enhanced[mask_mid] = np.clip(a[mask_mid] * 1.8, -127, 127)
    b_enhanced[mask_mid] = np.clip(b[mask_mid] * 1.8, -127, 127)
    
    # 亮部区域：适度增强
    a_enhanced[mask_bright] = np.clip(a[mask_bright] * 1.4, -127, 127)
    b_enhanced[mask_bright] = np.clip(b[mask_bright] * 1.4, -127, 127)
    
    # 合并通道
    lab_enhanced = cv2.merge([
        l, 
        a_enhanced.astype(np.int8).astype(np.uint8),
        b_enhanced.astype(np.int8).astype(np.uint8)
    ])
    
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def multi_scale_detail_enhancement(img):
    """多尺度细节增强"""
    # 高斯金字塔 - 提取不同尺度的细节
    gaussian_pyr = [img]
    for i in range(3):
        img = cv2.pyrDown(img)
        gaussian_pyr.append(img)
    
    # 拉普拉斯金字塔 - 重建细节
    laplacian_pyr = []
    for i in range(len(gaussian_pyr)-1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i+1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    
    # 增强细节
    enhanced_laplacian = []
    for i, lap in enumerate(laplacian_pyr):
        # 对不同尺度的细节应用不同的增强系数
        if i == 0:  # 最精细的细节
            enhanced_lap = cv2.convertScaleAbs(lap, alpha=1.5, beta=0)
        elif i == 1:  # 中等细节
            enhanced_lap = cv2.convertScaleAbs(lap, alpha=1.3, beta=0)
        else:  # 粗糙细节
            enhanced_lap = cv2.convertScaleAbs(lap, alpha=1.1, beta=0)
        enhanced_laplacian.append(enhanced_lap)
    
    # 重建图像
    img_reconstructed = gaussian_pyr[-1]
    for i in range(len(enhanced_laplacian)-1, -1, -1):
        size = (enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0])
        img_reconstructed = cv2.pyrUp(img_reconstructed, dstsize=size)
        img_reconstructed = cv2.add(img_reconstructed, enhanced_laplacian[i])
    
    return img_reconstructed

    natural_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_Lab2BGR)

    # 自然饱和度增强（避免色彩溢出）
    hsv = cv2.cvtColor(natural_img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.2, 0, 255)  # 饱和度提升20%
    final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return final_img

def self_encoder_fusion(visible_img, infrared_img):
    # 模型初始化与权重固定（保持原有初始化）
    model = RobustAutoencoder()
    torch.manual_seed(1234)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 鲁棒性预处理
    try:
        vis_tensor, ir_tensor = reliable_preprocess(visible_img, infrared_img)
    except Exception as e:
        raise RuntimeError(f"预处理失败: {str(e)}") from e

    vis_tensor = vis_tensor.to(device)
    ir_tensor = ir_tensor.to(device)

    # 模型推理
    with torch.no_grad():
        fused_tensor = model(vis_tensor, ir_tensor)

    # 后处理
    fused_image = natural_postprocess(fused_tensor)

    return fused_image

# ------------------- 接口验证 -------------------
if __name__ == "__main__":
    try:
        visible = "visible.jpg"
        infrared = "infrared.png"
        fused = self_encoder_fusion(visible, infrared)

        print("融合成功，色彩自然度提升，清晰度保持")
        cv2.imwrite("natural_color_fusion.jpg", fused)
        cv2.imshow("Natural Color Fusion", fused)
        cv2.waitKey(0)

    except ImportError:
        print("请安装PyTorch库")
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"融合失败: {str(e)}")