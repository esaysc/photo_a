import cv2
import numpy as np
import torch
from sklearn.decomposition import DictionaryLearning

def sparse_fusion(visible_img, infrared_img):
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将图像转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)

    # 降采样以减少计算量
    scale_factor = 0.25  # 降采样到1/4大小
    width = int(visible_img.shape[1] * scale_factor)
    height = int(visible_img.shape[0] * scale_factor)
    visible_img = cv2.resize(visible_img, (width, height))
    infrared_img = cv2.resize(infrared_img, (width, height))

    # 转换为PyTorch张量并移到GPU
    visible_tensor = torch.from_numpy(visible_img).float().to(device)
    infrared_tensor = torch.from_numpy(infrared_img).float().to(device)

    # 图像块大小和步长
    patch_size = 4
    stride = 4

    # 使用PyTorch的unfold操作提取图像块
    def extract_patches_gpu(img_tensor):
        patches = img_tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        patches = patches.contiguous().view(-1, patch_size * patch_size)
        return patches

    # 在GPU上提取图像块
    patches_vis = extract_patches_gpu(visible_tensor)
    patches_ir = extract_patches_gpu(infrared_tensor)

    # 将数据移回CPU进行字典学习（因为sklearn不支持GPU）
    patches_vis_cpu = patches_vis.cpu().numpy()
    patches_ir_cpu = patches_ir.cpu().numpy()

    # 字典学习（优化参数）
    n_components = 16  # 进一步减小字典大小
    dict_learner = DictionaryLearning(
        n_components=n_components,
        transform_algorithm='threshold',  # 使用更快的算法
        max_iter=10,                     # 减少迭代次数
        transform_alpha=0.2,             # 增加稀疏性
        n_jobs=-1                        # 使用所有CPU核心
    )
    
    # 字典学习和稀疏编码
    dictionary = dict_learner.fit(np.vstack([patches_vis_cpu, patches_ir_cpu])).components_
    codes_vis = dict_learner.transform(patches_vis_cpu)
    codes_ir = dict_learner.transform(patches_ir_cpu)

    # 将字典和编码移到GPU进行融合操作
    dictionary_gpu = torch.from_numpy(dictionary).float().to(device)
    codes_vis_gpu = torch.from_numpy(codes_vis).float().to(device)
    codes_ir_gpu = torch.from_numpy(codes_ir).float().to(device)

    # 在GPU上进行融合
    fused_codes = torch.where(torch.abs(codes_vis_gpu) > torch.abs(codes_ir_gpu),
                             codes_vis_gpu, codes_ir_gpu)

    # 在GPU上重建图像块
    fused_patches = torch.mm(fused_codes, dictionary_gpu)

    # 重建完整图像
    rows = (height - patch_size + 1) // stride
    cols = (width - patch_size + 1) // stride
    fused_image = torch.zeros((height, width), device=device)
    ones = torch.zeros((height, width), device=device)

    # 使用PyTorch的高效操作重建图像
    idx = 0
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = fused_patches[idx].view(patch_size, patch_size)
            fused_image[i:i+patch_size, j:j+patch_size] += patch
            ones[i:i+patch_size, j:j+patch_size] += 1
            idx += 1

    # 平均重叠区域
    fused_image = torch.div(fused_image, ones.clamp(min=1))

    # 将结果移回CPU并转换为numpy数组
    fused_image = fused_image.cpu().numpy()

    # 恢复原始尺寸
    fused_image = cv2.resize(fused_image, (int(width/scale_factor), int(height/scale_factor)))

    # 归一化处理
    fused_image = np.uint8(cv2.normalize(fused_image, None, 0, 255, cv2.NORM_MINMAX))
    return fused_image