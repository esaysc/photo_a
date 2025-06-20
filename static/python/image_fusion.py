import cv2
import numpy as np
import pywt
import argparse

def wavelet_fusion(visible_img, infrared_img):
    # 确保输入为彩色图像（3通道），若为灰度图则转换为3通道
    if len(visible_img.shape) == 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_GRAY2BGR)
    if len(infrared_img.shape) == 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)

    fused_channels = []
    for ch in range(3):  # 处理R、G、B三个通道
        vis_ch = visible_img[:, :, ch].astype(np.float32)
        ir_ch = infrared_img[:, :, ch].astype(np.float32)

        # 小波分解（使用db3小波，2层分解）
        coeffs1 = pywt.wavedec2(vis_ch, 'db3', level=2)
        coeffs2 = pywt.wavedec2(ir_ch, 'db3', level=2)

        # 低频分量取平均
        fused_approx = (coeffs1[0] + coeffs2[0]) * 0.5

        # 高频分量梯度加权融合
        fused_details = []
        for i in range(1, len(coeffs1)):  # 遍历各层高频分量
            ch_details = []
            for c1, c2 in zip(coeffs1[i], coeffs2[i]):  # 遍历每个方向（水平、垂直、对角线）
                # 计算梯度幅值
                grad1 = np.abs(cv2.Sobel(c1, cv2.CV_64F, 1, 1)) + np.abs(cv2.Sobel(c1, cv2.CV_64F, 0, 1))
                grad2 = np.abs(cv2.Sobel(c2, cv2.CV_64F, 1, 1)) + np.abs(cv2.Sobel(c2, cv2.CV_64F, 0, 1))
                # 梯度加权融合
                weight = grad1 / (grad1 + grad2 + 1e-8)
                fused = weight * c1 + (1 - weight) * c2
                ch_details.append(fused)
            fused_details.append(tuple(ch_details))

        # 重建通道
        fused_coeffs = [fused_approx] + fused_details
        fused_ch = pywt.waverec2(fused_coeffs, 'db3')

        # 非锐化掩膜增强清晰度
        blurred = cv2.GaussianBlur(fused_ch, (5, 5), 0)
        fused_ch = fused_ch + (fused_ch - blurred)
        fused_ch = np.clip(fused_ch, 0, 255).astype(np.uint8)
        fused_channels.append(fused_ch)

    fused_image = cv2.merge(fused_channels)
    return fused_image

def pyramid_fusion(visible_img, infrared_img):
    # 确保输入为彩色图像（3通道），若为灰度图则转换为3通道
    if len(visible_img.shape) == 2:
        visible_img = cv2.cvtColor(visible_img, cv2.COLOR_GRAY2BGR)
    if len(infrared_img.shape) == 2:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)

    levels = 4  # 增加金字塔层数至4层
    fused_channels = []

    for ch in range(3):  # 处理R、G、B三个通道
        vis_ch = visible_img[:, :, ch].astype(np.float32)
        ir_ch = infrared_img[:, :, ch].astype(np.float32)

        # 构建高斯金字塔
        gpA = [vis_ch.copy()]
        gpB = [ir_ch.copy()]
        for i in range(levels):
            gpA.append(cv2.pyrDown(gpA[-1]))
            gpB.append(cv2.pyrDown(gpB[-1]))

        # 构建拉普拉斯金字塔
        lpA = [gpA[levels]]
        lpB = [gpB[levels]]
        for i in range(levels, 0, -1):
            size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
            GE = cv2.pyrUp(gpA[i], dstsize=size)
            L = gpA[i-1] - GE
            lpA.append(L)
            GE = cv2.pyrUp(gpB[i], dstsize=size)
            L = gpB[i-1] - GE
            lpB.append(L)

        # 基于方差的高频加权融合
        LS = []
        for la, lb in zip(lpA, lpB):
            if la.size == 0 or lb.size == 0:
                LS.append(lb if la.size == 0 else la)
                continue
            var_a = cv2.blur(la**2, (3, 3)) - cv2.blur(la, (3, 3))**2
            var_b = cv2.blur(lb**2, (3, 3)) - cv2.blur(lb, (3, 3))**2
            weight = var_a / (var_a + var_b + 1e-8)
            fused = weight * la + (1 - weight) * lb
            LS.append(fused)

        # 重建通道
        ls_ = LS[0]
        for i in range(1, levels+1):
            size = (LS[i].shape[1], LS[i].shape[0])
            ls_ = cv2.pyrUp(ls_, dstsize=size)
            ls_ = ls_ + LS[i]
        ls_ = np.clip(ls_, 0, 255).astype(np.uint8)
        fused_channels.append(ls_)

    # 轮廓增强（叠加边缘信息）
    fused_image = cv2.merge(fused_channels)
    gray = cv2.cvtColor(fused_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    fused_image = cv2.addWeighted(fused_image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

    return fused_image

def main():
    parser = argparse.ArgumentParser(description='Color Image Fusion with Enhanced Sharpness')
    parser.add_argument('--visible', required=True, help='Path to visible light image (color or grayscale)')
    parser.add_argument('--infrared', required=True, help='Path to infrared image (color or grayscale)')
    parser.add_argument('--output', required=True, help='Path to output fused color image')
    parser.add_argument('--method', required=True, choices=['fast', 'wavelet', 'pyramid', 'sparse', 'cnn', 'self-encoder', 'gan', 'ganresnet'],
                      help='Fusion method (wavelet/pyramid)')
    parser.add_argument('--model', type=str, default='cnn_fusion_model.pth', help='模型路径')

    args = parser.parse_args()

    # 读取图像
    visible_img = cv2.imread(args.visible)
    infrared_img = cv2.imread(args.infrared)

    if visible_img is None or infrared_img is None:
        raise ValueError("无法读取输入图像")

    # 统一尺寸
    if visible_img.shape[:2] != infrared_img.shape[:2]:
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
    # elif args.method == 'cnn':
    #     try:
    #         from convolutional_neural_network import convolutional_neural_network
    #         fused_image = convolutional_neural_network(visible_img, infrared_img,args.model)
    #     except ImportError:
    #         raise ImportError("卷积神经网络需要安装scikit-learn库并确保convolutional_neural_network.py在同一目录下")
    elif args.method == 'cnn':
        try:
            from cnn_fusion import cnn_fusion
            fused_image = cnn_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("CNN融合方法需要安装PyTorch库并确保cnn_fusion.py在同一目录下")
    elif args.method == 'self-encoder':
        try:
            from self_encoder_fusion import self_encoder_fusion
            # fused_image = wavelet_fusion(visible_img, infrared_img)
            fused_image = self_encoder_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("自编码器融合方法需要安装PyTorch库并确保self_encoder_fusion.py在同一目录下")
    elif args.method == 'gan':
        try:
            from gan_fusion import gan_fusion
            fused_image = gan_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("GAN融合方法需要安装PyTorch库并确保gan_fusion.py在同一目录下")
    elif args.method == 'ganresnet':
        try:
            from ganresnet_fusion import ganresnet_fusion
            # fused_image = pyramid_fusion(visible_img, infrared_img)
            fused_image = ganresnet_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("GAN-ResNet融合方法需要安装PyTorch库并确保ganresnet_fusion.py在同一目录下")
    elif args.method == 'best':
        try:
            from best_fusion import best_fusion
            fused_image = best_fusion(visible_img, infrared_img)
            # fused_image = ganresnet_fusion(visible_img, infrared_img)
        except ImportError:
            raise ImportError("best融合方法需要安装PyTorch库并确保ganresnet_fusion.py在同一目录下")
    else:
        raise ValueError("无效的融合方法")
    # 保存结果
    cv2.imwrite(args.output, fused_image)


if __name__ == '__main__':
    main()