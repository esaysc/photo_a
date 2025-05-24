import cv2
import numpy as np
import pywt
import argparse

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

def main():
    parser = argparse.ArgumentParser(description='Image Fusion using Multiple Methods')
    parser.add_argument('--visible', required=True, help='Path to visible light image')
    parser.add_argument('--infrared', required=True, help='Path to infrared image')
    parser.add_argument('--output', required=True, help='Path to output fused image')
    parser.add_argument('--method', required=True, choices=['wavelet', 'pyramid', 'sparse', 'cnn', 'self-encoder', 'gan'], help='Fusion method')
    
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
        try:
            from sparse_fusion import sparse_fusion
            fused_image = sparse_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("稀疏表示法需要安装scikit-learn库并确保sparse_fusion.py在同一目录下")
    elif args.method == 'cnn':
        try:
            from cnn_fusion import cnn_fusion
            fused_image = cnn_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("CNN融合方法需要安装PyTorch库并确保cnn_fusion.py在同一目录下")
    elif args.method == 'self-encoder':
        try:
            from self_encoder_fusion import self_encoder_fusion
            fused_image = self_encoder_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("自编码器融合方法需要安装PyTorch库并确保self_encoder_fusion.py在同一目录下")
    elif args.method == 'gan':
        try:
            from gan_fusion import gan_fusion
            fused_image = gan_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("GAN融合方法需要安装PyTorch库并确保gan_fusion.py在同一目录下")
    
    # 保存结果
    cv2.imwrite(args.output, fused_image)

if __name__ == '__main__':
    main()
