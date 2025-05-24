package com.ruoyi.cms.service;

public interface IImageFusionService {
    /**
     * 执行图像融合操作
     * @param visibleImage 可见光图像路径
     * @param infraredImage 红外线图像路径
     * @param fusionMethod 融合方法（"wavelet"或"pyramid"）
     * @return 融合结果图像路径
     */
    String fusionImages(String visibleImage, String infraredImage, String fusionMethod);
}