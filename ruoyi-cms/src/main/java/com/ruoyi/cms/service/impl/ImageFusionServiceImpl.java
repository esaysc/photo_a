package com.ruoyi.cms.service.impl;

import com.ruoyi.cms.service.IImageFusionService;
import com.ruoyi.common.config.RuoYiConfig;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.io.File;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class ImageFusionServiceImpl implements IImageFusionService {
    private static final Logger log = LoggerFactory.getLogger(ImageFusionServiceImpl.class);
    
    private final String pythonPath = Paths.get(System.getProperty("user.dir"), "venv/Scripts/python.exe").toString();  // 使用虚拟环境的Python
    // private final String scriptPath = Paths.get(RuoYiConfig.getProfile(), "python/image_fusion.py").toString();
    private final String scriptPath = Paths.get(RuoYiConfig.getProfile(), "python", "image_fusion.py").toString();

    // 添加算法计数器，线程安全
    private static final ConcurrentHashMap<String, AtomicInteger> algorithmCounters = new ConcurrentHashMap<>();

    @Override
    public String fusionImages(String visibleImage, String infraredImage, String fusionMethod) {
        try {
            log.info("接收到的图片路径: visible={}, infrared={}", visibleImage, infraredImage);
            
            // 获取实际的文件系统路径
            String visiblePath = Paths.get(RuoYiConfig.getProfile(), visibleImage).toString();
            String infraredPath = Paths.get(RuoYiConfig.getProfile(), infraredImage).toString();
            
            log.info("处理后的完整路径: visible={}, infrared={}", visiblePath, infraredPath);
            
            // 生成按算法名+次数命名的输出文件名
            String outputFileName = generateOutputFileName(fusionMethod);
            String outputPath = Paths.get(RuoYiConfig.getProfile(), "fusion", outputFileName).toString();
            
            log.info("生成的输出文件名: {}", outputPath);
            
            // 确保输出目录存在
            new File(outputPath).getParentFile().mkdirs();

            ProcessBuilder processBuilder = new ProcessBuilder(
                pythonPath,
                scriptPath,
                "--visible", visiblePath,
                "--infrared", infraredPath,
                "--method", fusionMethod,
                "--output", outputPath
            );
            
            // 合并错误流到标准输出流
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            // 读取Python程序的输出
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                // 返回可访问的路径，添加 /profile 前缀
                String resultPath = "/profile/fusion/" + outputFileName;
                log.info("执行成功: {}", resultPath);
                return resultPath;
            } else {
                log.error("Python处理失败: {}", output.toString());
                throw new RuntimeException("Python处理失败: " + output.toString());
            }
        } catch (Exception e) {
            log.error("图像融合处理失败", e);
            throw new RuntimeException("图像融合处理失败: " + e.getMessage());
        }
    }

    /**
     * 生成按算法名+次数命名的输出文件名
     * @param fusionMethod 融合算法名称
     * @return 文件名，格式如：wavelet_001.jpg
     */
    private String generateOutputFileName(String fusionMethod) {
        // 获取或创建该算法的计数器
        AtomicInteger counter = algorithmCounters.computeIfAbsent(fusionMethod, k -> new AtomicInteger(0));
        
        // 自增计数
        int count = counter.incrementAndGet();
        
        // 格式化文件名：算法名_序号.jpg
        return String.format("%s_%03d.jpg", fusionMethod, count);
    }

    /**
     * 重置指定算法的计数器（可选方法，用于管理）
     * @param fusionMethod 算法名称
     */
    public void resetCounter(String fusionMethod) {
        algorithmCounters.put(fusionMethod, new AtomicInteger(0));
        log.info("已重置算法 {} 的计数器", fusionMethod);
    }

    /**
     * 获取指定算法的当前计数（可选方法，用于查询）
     * @param fusionMethod 算法名称
     * @return 当前计数
     */
    public int getCurrentCount(String fusionMethod) {
        AtomicInteger counter = algorithmCounters.get(fusionMethod);
        return counter != null ? counter.get() : 0;
    }
}