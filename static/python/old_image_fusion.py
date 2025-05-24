import cv2
import numpy as np
import pywt
import argparse
from sklearn.decomposition import DictionaryLearning

def wavelet_fusion(visible_img, infrared_img):
    # 将图像转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # 进行小波分解
    coeffs1 = pywt.wavedec2(visible_img, 'db1', level=1)
    coeffs2 = pywt.wavedec2(infrared_img, 'db1', level=1)
    
    # 融合规则
    # 低频分量取平均
    fused_approximation = (coeffs1[0] + coeffs2[0]) * 0.5
    
    # 高频分量取最大值
    fused_details = []
    for i in range(1, len(coeffs1)):
        fused_details.append(tuple(np.maximum(c1, c2) for c1, c2 in zip(coeffs1[i], coeffs2[i])))
    
    # 重建融合后的图像
    fused_coeffs = [fused_approximation] + fused_details
    fused_image = pywt.waverec2(fused_coeffs, 'db1')
    
    # 归一化处理
    fused_image = np.uint8(cv2.normalize(fused_image, None, 0, 255, cv2.NORM_MINMAX))
    
    return fused_image

def pyramid_fusion(visible_img, infrared_img):
    # 将图像转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # 构建拉普拉斯金字塔
    levels = 3
    G1 = visible_img.copy()
    G2 = infrared_img.copy()
    gpA = [G1]
    gpB = [G2]
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        gpA.append(G1)
        gpB.append(G2)
    
    # 构建拉普拉斯金字塔
    lpA = [gpA[levels-1]]
    lpB = [gpB[levels-1]]
    for i in range(levels-1, 0, -1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize=size)
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)
        GE = cv2.pyrUp(gpB[i], dstsize=size)
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)
    
    # 融合金字塔
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols = la.shape
        ls = np.zeros((rows, cols), dtype=np.float32)
        ls = la * 0.5 + lb * 0.5
        LS.append(ls)
    
    # 重建图像
    ls_ = LS[0]
    for i in range(1, levels):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])
    
    # 归一化处理
    ls_ = np.uint8(cv2.normalize(ls_, None, 0, 255, cv2.NORM_MINMAX))
    return ls_

def sparse_fusion(visible_img, infrared_img):
    # 将图像转换为灰度图
    if len(visible_img.shape) > 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_BGR2GRAY)
    if len(infrared_img.shape) > 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # 图像块大小和步长
    patch_size = 8
    stride = 4
    
    # 提取图像块
    def extract_patches(img):
        patches = []
        for i in range(0, img.shape[0] - patch_size + 1, stride):
            for j in range(0, img.shape[1] - patch_size + 1, stride):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        return np.array(patches)
    
    # 提取两个图像的图像块
    patches_vis = extract_patches(visible_img)
    patches_ir = extract_patches(infrared_img)
    
    # 字典学习
    n_components = 100
    dict_learner = DictionaryLearning(n_components=n_components, transform_algorithm='lasso_lars')
    dictionary = dict_learner.fit(np.vstack([patches_vis, patches_ir])).components_
    
    # 稀疏编码
    def sparse_encode(patches, dictionary):
        return dict_learner.transform(patches)
    
    codes_vis = sparse_encode(patches_vis, dictionary)
    codes_ir = sparse_encode(patches_ir, dictionary)
    
    # 融合规则：选择稀疏系数较大的一个
    fused_codes = np.where(np.abs(codes_vis) > np.abs(codes_ir), codes_vis, codes_ir)
    
    # 重建图像块
    fused_patches = np.dot(fused_codes, dictionary)
    
    # 重建完整图像
    fused_image = np.zeros_like(visible_img, dtype=np.float32)
    count = np.zeros_like(visible_img, dtype=np.float32)
    patch_idx = 0
    
    for i in range(0, visible_img.shape[0] - patch_size + 1, stride):
        for j in range(0, visible_img.shape[1] - patch_size + 1, stride):
            patch = fused_patches[patch_idx].reshape(patch_size, patch_size)
            fused_image[i:i+patch_size, j:j+patch_size] += patch
            count[i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1
    
    # 平均重叠区域
    fused_image = np.divide(fused_image, count, where=count!=0)
    
    # 归一化处理
    fused_image = np.uint8(cv2.normalize(fused_image, None, 0, 255, cv2.NORM_MINMAX))
    return fused_image

def main():
    parser = argparse.ArgumentParser(description='Image Fusion using Multiple Methods')
    parser.add_argument('--visible', required=True, help='Path to visible light image')
    parser.add_argument('--infrared', required=True, help='Path to infrared image')
    parser.add_argument('--output', required=True, help='Path to output fused image')
    parser.add_argument('--method', required=True, choices=['wavelet', 'pyramid', 'sparse'], help='Fusion method')
    
    args = parser.parse_args()
    
    # 读取图像
    visible_img = cv2.imread(args.visible)
    infrared_img = cv2.imread(args.infrared)
    
    if visible_img is None or infrared_img is None:
        raise ValueError("无法读取输入图像")
    
    # 确保两张图片尺寸相同
    if visible_img.shape != infrared_img.shape:
        infrared_img = cv2.resize(infrared_img, (visible_img.shape[1], visible_img.shape[0]))
    
    # 根据选择的方法进行图像融合
    if args.method == 'wavelet':
        fused_image = wavelet_fusion(visible_img, infrared_img)
    elif args.method == 'pyramid':
        fused_image = pyramid_fusion(visible_img, infrared_img)
    elif args.method == 'sparse':
        fused_image = sparse_fusion(visible_img, infrared_img)
    
    
    # 保存结果
    cv2.imwrite(args.output, fused_image)

if __name__ == '__main__':
    main()