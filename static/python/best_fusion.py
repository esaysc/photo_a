import cv2
import numpy as np
from image_fusion import ImageFusion
from sklearn.metrics import peak_signal_noise_ratio, structural_similarity

class BestFusion:
    def __init__(self):
        self.fusion = ImageFusion()
        self.methods = [
            'cnn',
            'gan',
            'sparse',
            'self-encoder'
        ]
    
    def evaluate_fusion(self, fused_img, visible_img, infrared_img):
        # 计算与可见光和红外图像的PSNR
        psnr_visible = peak_signal_noise_ratio(visible_img, fused_img)
        psnr_infrared = peak_signal_noise_ratio(infrared_img, fused_img)
        
        # 计算结构相似性
        ssim_visible = structural_similarity(visible_img, fused_img)
        ssim_infrared = structural_similarity(infrared_img, fused_img)
        
        # 综合评分
        score = (psnr_visible + psnr_infrared) * 0.5 + (ssim_visible + ssim_infrared) * 0.5
        return score
    
    def find_best_method(self, visible_path, infrared_path):
        best_score = -float('inf')
        best_method = None
        best_result = None
        
        # 读取原始图像
        visible_img = cv2.imread(visible_path, cv2.IMREAD_GRAYSCALE)
        infrared_img = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)
        
        # 尝试每种方法
        for method in self.methods:
            try:
                # 执行融合
                fused_img = self.fusion.fuse_images(visible_path, infrared_path, method)
                
                # 评估结果
                score = self.evaluate_fusion(fused_img, visible_img, infrared_img)
                
                if score > best_score:
                    best_score = score
                    best_method = method
                    best_result = fused_img
                    
            except Exception as e:
                print(f"Method {method} failed: {str(e)}")
                continue
        
        return best_method, best_result, best_score

    def process(self, visible_path, infrared_path, save_path):
        # 找到最佳方法并融合
        best_method, fused_img, score = self.find_best_method(visible_path, infrared_path)
        
        if best_method and fused_img is not None:
            # 保存结果
            cv2.imwrite(save_path, fused_img)
            return {
                'method': best_method,
                'score': score,
                'path': save_path
            }
        else:
            raise Exception("No suitable fusion method found")