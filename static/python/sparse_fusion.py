import cv2
import numpy as np
import torch
from sklearn.decomposition import DictionaryLearning

def sparse_fusion(visible_img, infrared_img):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输入图像预处理（强制彩色）
    def preprocess_color(img, is_ir=False):
        if len(img.shape) == 2:
            if is_ir:
                # 红外灰度图转伪彩色（保持与可见光相似的色彩空间）
                img = cv2.applyColorMap(img, cv2.COLOR_MAP_JET)  # 转为3通道伪彩色
            else:
                # 可见光灰度图转RGB（保留亮度信息）
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # 确保为3通道（截断Alpha通道等）
            img = img[:, :, :3]
        return img

    # 预处理可见光和红外图像
    visible_color = preprocess_color(visible_img)
    infrared_color = preprocess_color(infrared_img, is_ir=True)
    
    # 统一尺寸
    h, w = visible_color.shape[:2]
    infrared_color = cv2.resize(infrared_color, (w, h))
    
    # 降采样（平衡速度与质量）
    scale_factor = 0.5  # 降采样至1/2尺寸（原代码0.25可能导致细节丢失）
    nh, nw = int(h * scale_factor), int(w * scale_factor)
    visible_down = cv2.resize(visible_color, (nw, nh))
    infrared_down = cv2.resize(infrared_color, (nw, nh))
    
    # 分离RGB通道（分别处理每个颜色通道）
    v_channels = cv2.split(visible_down)  # [R, G, B]
    ir_channels = cv2.split(infrared_down)  # [R', G', B']（伪彩色通道）
    fused_channels = []
    
    # 稀疏融合核心逻辑（每个通道独立处理）
    patch_size, stride = 8, 4  # 增大patch尺寸以捕获更多色彩细节
    n_components = 32  # 字典大小适中，避免过拟合
    
    for v_ch, ir_ch in zip(v_channels, ir_channels):
        # 转换为GPU张量
        v_tensor = torch.from_numpy(v_ch).float().to(device)
        ir_tensor = torch.from_numpy(ir_ch).float().to(device)
        
        # 提取图像块并归一化（避免通道间干扰）
        def extract_and_normalize(tensor):
            patches = tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
            patches = patches.reshape(-1, patch_size * patch_size)
            mean = patches.mean(dim=1, keepdim=True)
            std = patches.std(dim=1, keepdim=True).clamp(min=1e-6)
            return (patches - mean) / std, mean, std, patches.shape[0]
        
        # 提取可见光和红外图像块
        (patches_v, mean_v, std_v, n_patches), \
        (patches_ir, mean_ir, std_ir, _) = map(extract_and_normalize, [v_tensor, ir_tensor])
        
        # 字典学习（CPU执行）
        dict_learner = DictionaryLearning(
            n_components=n_components,
            transform_algorithm='lasso_lars',  # 更精确的编码算法
            max_iter=100,                       # 增加迭代次数提升字典质量
            transform_alpha=0.4,                # 平衡稀疏性与重构误差
            n_jobs=-1
        )
        # 联合训练字典（融合可见光与红外特征）
        all_patches = np.vstack([patches_v.cpu().numpy(), patches_ir.cpu().numpy()])
        dictionary = dict_learner.fit(all_patches).components_
        
        # 稀疏编码（获取系数）
        codes_v = dict_learner.transform(patches_v.cpu().numpy())
        codes_ir = dict_learner.transform(patches_ir.cpu().numpy())
        
        # 融合策略：保留可见光主导的系数（确保色彩一致）
        codes_v_gpu = torch.from_numpy(codes_v).float().to(device)
        codes_ir_gpu = torch.from_numpy(codes_ir).float().to(device)
        # 计算系数幅度，优先选择可见光的显著特征
        fused_codes = torch.where(
            torch.abs(codes_v_gpu) > torch.abs(codes_ir_gpu),
            codes_v_gpu, 
            codes_ir_gpu * 0.5  # 红外系数衰减，避免色彩偏移
        )
        
        # 重建图像块（反归一化）
        dictionary_gpu = torch.from_numpy(dictionary).float().to(device)
        patches_recon = torch.mm(fused_codes, dictionary_gpu) * std_v + mean_v
        
        # 拼接图像块（处理重叠区域）
        fused_ch = torch.zeros((nh, nw), device=device)
        count = torch.zeros((nh, nw), device=device)
        idx = 0
        for i in range(0, nh - patch_size + 1, stride):
            for j in range(0, nw - patch_size + 1, stride):
                patch = patches_recon[idx].view(patch_size, patch_size)
                fused_ch[i:i+patch_size, j:j+patch_size] += patch
                count[i:i+patch_size, j:j+patch_size] += 1
                idx += 1
        fused_ch = fused_ch / count.clamp(min=1)  # 平均重叠区域
        
        # 恢复尺寸并转换为numpy
        fused_ch = cv2.resize(fused_ch.cpu().numpy(), (w, h))
        fused_channels.append(fused_ch)
    
    # 合并RGB通道（确保与可见光色彩一致）
    fused_image = cv2.merge(fused_channels)
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    
    # 色彩校准（匹配可见光色域）
    # 计算可见光均值作为色彩参考
    vis_lab = cv2.cvtColor(visible_color, cv2.COLOR_BGR2Lab)
    fused_lab = cv2.cvtColor(fused_image, cv2.COLOR_BGR2Lab)
    fused_lab[..., 1:] = vis_lab[..., 1:]  # 强制匹配可见光的a/b通道（色度）
    fused_image = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
    
    return fused_image

# ------------------- 示例用法 -------------------
if __name__ == "__main__":
    # 读取图像（可见光彩色，红外灰度）
    visible_img = cv2.imread("visible.jpg")       # 可见光彩色图像
    infrared_img = cv2.imread("infrared.png", 0)  # 红外灰度图像
    
    # 执行彩色融合
    fused_result = sparse_fusion(visible_img, infrared_img)
    
    # 保存与显示
    cv2.imwrite("fused_color_sparse.jpg", fused_result)
    cv2.imshow("Color-Sparse Fusion Result", fused_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()