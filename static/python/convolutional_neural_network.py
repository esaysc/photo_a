import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pywt
from sklearn.decomposition import DictionaryLearning
import argparse

class ConvFusionNet(nn.Module):
    """增强型卷积融合网络"""
    def __init__(self):
        super(ConvFusionNet, self).__init__()
        
        # 特征提取层 - 可见光
        self.visible_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.visible_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.visible_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 特征提取层 - 红外
        self.ir_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.ir_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ir_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # 重建层 - 明确输出3通道彩色图像
        self.recon_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.recon_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.recon_conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)  # 确保输出3通道
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, visible, infrared):
        # 可见光特征提取
        v_feat1 = self.relu(self.visible_conv1(visible))
        v_feat2 = self.relu(self.visible_conv2(v_feat1))
        v_feat3 = self.relu(self.visible_conv3(v_feat2))
        
        # 红外特征提取
        i_feat1 = self.relu(self.ir_conv1(infrared))
        i_feat2 = self.relu(self.ir_conv2(i_feat1))
        i_feat3 = self.relu(self.ir_conv3(i_feat2))
        
        # 特征融合
        concat_feat = torch.cat([v_feat3, i_feat3], dim=1)
        
        # 注意力机制
        attn_weights = self.attention(concat_feat)
        weighted_feat = concat_feat * attn_weights
        
        # 融合处理
        fused_feat1 = self.relu(self.fusion_conv1(weighted_feat))
        fused_feat2 = self.relu(self.fusion_conv2(fused_feat1))
        
        # 图像重建 - 确保输出3通道彩色
        recon_feat1 = self.relu(self.recon_conv1(fused_feat2))
        recon_feat2 = self.relu(self.recon_conv2(recon_feat1))
        output = torch.tanh(self.recon_conv3(recon_feat2))
        
        return output

def preprocess_image(img, is_ir=False, target_size=None):
    """改进的图像预处理：确保彩色输入和优化的色彩空间对齐"""
    if target_size:
        img = cv2.resize(img, target_size)
    
    # 处理单通道图像，确保输入为3通道
    if len(img.shape) == 2:
        if is_ir:
            # 红外图像转为伪彩色，使用改进的JET色彩映射
            img = cv2.applyColorMap(img, cv2.COLOR_MAP_JET)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[..., 1] = np.clip(img[..., 1] * 0.8, 0, 255)  # 适度降低饱和度
            img[..., 2] = np.clip(img[..., 2] * 1.1, 0, 255)  # 增加亮度
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            # 可见光灰度图转为3通道RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 色彩空间标准化 - 使用LAB空间进行亮度调整
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    # 亮度增强 - 对红外和可见光采用不同策略
    if is_ir:
        # 红外图像：全局直方图均衡化增强热特征
        l = cv2.equalizeHist(l)
    else:
        # 可见光图像：自适应直方图均衡化保留细节
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
    
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    
    # 归一化到[-1, 1]范围
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    # 转换为PyTorch张量
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    
    return img_tensor

def postprocess_image(tensor):
    """改进的图像后处理：增强色彩自然度和细节"""
    # 转回numpy数组
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # 转回[0, 255]范围
    img = ((img + 1.0) * 127.5).astype(np.uint8)
    
    # 色彩增强 - 在LAB空间进行处理
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    # 自适应对比度增强
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 色彩平衡调整
    a = np.clip(a * 1.05, 0, 255).astype(np.uint8)  # 微调绿色-红色通道
    b = np.clip(b * 1.05, 0, 255).astype(np.uint8)  # 微调蓝色-黄色通道
    
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    
    # 细节增强
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)  # 锐化处理
    
    return img

def load_model(model_path="cnn_fusion_model.pth"):
    """加载预训练模型"""
    model = ConvFusionNet()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"已加载预训练模型: {model_path}")
    else:
        print("警告: 未找到预训练模型，将使用随机初始化权重")
    
    model.eval()
    return model

def color_correction(img):
    """改进的自适应色彩校正，增强彩色输出效果"""
    # 转换到HSV空间进行色彩调整
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 自适应饱和度增强 - 暗部减少增强，亮部适当增强
    s_factor = np.clip(v / 255.0 * 0.3 + 0.9, 0.8, 1.2)  # 微调饱和度增强曲线
    s = np.clip(s * s_factor, 0, 255).astype(np.uint8)
    
    # 自适应亮度调整 - 保留更多原始亮度信息
    v_mean = np.mean(v)
    if v_mean < 90:  # 仅对较暗图像增强亮度
        v = np.clip(v * 1.15, 0, 255).astype(np.uint8)
    
    # 合并通道
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 最终色彩平衡 - 在YCrCb空间调整
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # 对比度增强
    y = cv2.addWeighted(y, 1.2, y, 0, -25)  # 微调对比度参数
    y = np.clip(y, 0, 255).astype(np.uint8)
    
    # 肤色保护 - 调整Cr通道
    cr = np.clip(cr * 0.95, 0, 255).astype(np.uint8)
    
    ycrcb = cv2.merge([y, cr, cb])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return img

def multi_scale_detail_fusion(visible, infrared, fused):
    """改进的多尺度细节融合，增强图像清晰度同时保留色彩"""
    # 确保输入都是彩色图像
    if len(visible.shape) == 2:
        visible = cv2.cvtColor(visible, cv2.COLOR_GRAY2BGR)
    if len(infrared.shape) == 2:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_GRAY2BGR)
    
    # 转换到YCbCr空间 - 分离亮度和色度
    visible_ycrcb = cv2.cvtColor(visible, cv2.COLOR_BGR2YCrCb)
    infrared_ycrcb = cv2.cvtColor(infrared, cv2.COLOR_BGR2YCrCb)
    fused_ycrcb = cv2.cvtColor(fused, cv2.COLOR_BGR2YCrCb)
    
    # 分离亮度和色度
    v_y, v_cr, v_cb = cv2.split(visible_ycrcb)
    i_y, i_cr, i_cb = cv2.split(infrared_ycrcb)
    f_y, f_cr, f_cb = cv2.split(fused_ycrcb)
    
    # 小波分解 - 对亮度通道进行多尺度分析
    v_coeffs = pywt.wavedec2(v_y, 'db1', level=2)
    i_coeffs = pywt.wavedec2(i_y, 'db1', level=2)
    f_coeffs = pywt.wavedec2(f_y, 'db1', level=2)
    
    # 融合策略：低频部分保留可见光，高频部分融合细节
    fused_coeffs = [v_coeffs[0]]  # 低频部分使用可见光的结构信息
    
    for i in range(1, len(v_coeffs)):
        # 对每个高频子带进行处理
        v_dh, v_dv, v_dd = v_coeffs[i]
        i_dh, i_dv, i_dd = i_coeffs[i]
        f_dh, f_dv, f_dd = f_coeffs[i]
        
        # 计算活动度测量 - 基于能量
        v_energy = v_dh**2 + v_dv**2 + v_dd**2
        i_energy = i_dh**2 + i_dv**2 + i_dd**2
        
        # 自适应权重融合 - 根据能量比例确定权重
        weight = np.clip(v_energy / (v_energy + i_energy + 1e-8), 0.4, 0.6)
        
        # 融合高频细节 - 结合可见光纹理和红外热特征
        dh_fused = weight * v_dh + (1 - weight) * i_dh
        dv_fused = weight * v_dv + (1 - weight) * i_dv
        dd_fused = weight * v_dd + (1 - weight) * i_dd
        
        fused_coeffs.append((dh_fused, dv_fused, dd_fused))
    
    # 小波重构
    fused_y = pywt.waverec2(fused_coeffs, 'db1')
    fused_y = fused_y[:f_cr.shape[0], :f_cr.shape[1]].astype(np.uint8)
    
    # 合并通道 - 使用融合后的亮度和原始融合图像的色度
    fused_ycrcb = cv2.merge([fused_y, f_cr, f_cb])
    fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return fused_bgr

def convolutional_neural_network(visible_img, infrared_img, model_path="cnn_fusion_model.pth"):
    """基于卷积神经网络的图像融合 - 确保输出彩色图像"""
    # 确保输入图像尺寸一致
    if visible_img.shape != infrared_img.shape:
        infrared_img = cv2.resize(infrared_img, (visible_img.shape[1], visible_img.shape[0]))
    
    # 保存原始尺寸
    original_size = visible_img.shape[:2]
    
    # 加载模型
    model = load_model(model_path)
    
    # 预处理图像 - 确保输入为彩色
    visible_tensor = preprocess_image(visible_img, is_ir=False)
    infrared_tensor = preprocess_image(infrared_img, is_ir=True)
    
    # 模型推理
    with torch.no_grad():
        fused_tensor = model(visible_tensor, infrared_tensor)
    
    # 后处理 - 增强色彩和细节
    fused_img = postprocess_image(fused_tensor)
    
    # 多尺度细节融合 - 结合可见光纹理和红外热特征
    detail_fused_img = multi_scale_detail_fusion(visible_img, infrared_img, fused_img)
    
    # 色彩校正 - 进一步优化彩色输出
    color_corrected_img = color_correction(detail_fused_img)
    
    # 确保最终输出尺寸与输入一致
    if color_corrected_img.shape[:2] != original_size:
        color_corrected_img = cv2.resize(color_corrected_img, (original_size[1], original_size[0]))
    
    return color_corrected_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于卷积神经网络的图像融合')
    parser.add_argument('--visible', type=str, required=True, help='可见光图像路径')
    parser.add_argument('--infrared', type=str, required=True, help='红外图像路径')
    parser.add_argument('--output', type=str, default='fused_cnn.jpg', help='输出图像路径')
    parser.add_argument('--model', type=str, default='cnn_fusion_model.pth', help='模型路径')
    args = parser.parse_args()
    
    # 读取图像
    visible_img = cv2.imread(args.visible)
    infrared_img = cv2.imread(args.infrared, 0)  # 假设红外图像是灰度图
    
    if visible_img is None or infrared_img is None:
        print("错误：无法读取输入图像")
        exit(1)
    
    # 执行融合
    fused_image = convolutional_neural_network(visible_img, infrared_img, args.model)
    
    # 保存结果 - 确保保存为彩色图像
    cv2.imwrite(args.output, fused_image)
    print(f"融合图像已保存至: {args.output}")