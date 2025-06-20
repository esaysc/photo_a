package com.ruoyi.cms.controller;


import com.ruoyi.cms.domain.TaskInfo;
import com.ruoyi.cms.service.IImageFusionService;
import com.ruoyi.cms.service.TaskInfoService;
import com.ruoyi.cms.service.impl.ImageFusionServiceImpl;
import com.ruoyi.common.core.domain.AjaxResult;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

@RestController
@RequestMapping("/cms/fusion")
public class FusionController {

    @Autowired
    private IImageFusionService fusionService;

    @Autowired
    private ImageFusionServiceImpl fusionServiceImpl;

    @Autowired
    private TaskInfoService taskInfoService;

    private final ExecutorService executorService = Executors.newFixedThreadPool(5);

    @PostMapping("/process")
    public AjaxResult processFusion(@RequestBody FusionRequest request) {
        String taskId = UUID.randomUUID().toString();

        TaskInfo taskInfo = new TaskInfo();
        taskInfo.setTaskId(taskId);
        taskInfo.setStatus("processing");
        taskInfo.setProgress(0);
        taskInfo.setCreateTime(new Date());
        taskInfoService.saveTask(taskInfo);

        // 使用自定义线程池执行异步任务
        CompletableFuture.runAsync(() -> {
            try {
                taskInfo.setStatus("processing");
                taskInfoService.updateTask(taskInfo);

                String visibleImage = request.getVisibleImage();
                String infraredImage = request.getInfraredImage();
                String method = request.getFusionMethod();

                taskInfo.setProgress(20);
                taskInfoService.updateTask(taskInfo);

                String result = fusionService.fusionImages(visibleImage, infraredImage, method);

                taskInfo.setProgress(80);
                taskInfoService.updateTask(taskInfo);

                if (result != null) {
                    taskInfo.setStatus("completed");
                    taskInfo.setProgress(100);
                    taskInfo.setResult(result);
                } else {
                    taskInfo.setStatus("failed");
                    taskInfo.setErrorMsg("融合失败");
                }

            } catch (Exception e) {
                taskInfo.setStatus("failed");
                taskInfo.setErrorMsg(e.getMessage());
            } finally {
                taskInfoService.updateTask(taskInfo);
            }
        }, executorService);  // 指定使用自定义线程池

        return AjaxResult.success("提交成功", Map.of("taskId", taskId));
    }


    // 添加进度查询接口
    @GetMapping("/progress/{taskId}")
    public AjaxResult getProgress(@PathVariable String taskId) {
        TaskInfo taskInfo = taskInfoService.getTask(taskId);
        if (taskInfo == null) {
            return AjaxResult.error("任务不存在");
        }

        Map<String, Object> result = new HashMap<>();
        result.put("taskId", taskId);
        result.put("status", taskInfo.getStatus());
        result.put("progress", taskInfo.getProgress());

        if ("completed".equals(taskInfo.getStatus())) {
            result.put("result", taskInfo.getResult());
        } else if ("failed".equals(taskInfo.getStatus())) {
            result.put("error", taskInfo.getErrorMsg());
        }

        return AjaxResult.success(result);
    }

    // 获取算法参数接口
    @GetMapping("/params/{fusionMethod}")
    public AjaxResult getAlgorithmParams(@PathVariable String fusionMethod) {
        List<Map<String, Object>> params = getParametersByMethod(fusionMethod);
        return AjaxResult.success(params);
    }

    // 获取算法执行计数
    @GetMapping("/count/{fusionMethod}")
    public AjaxResult getAlgorithmCount(@PathVariable String fusionMethod) {
        int count = fusionServiceImpl.getCurrentCount(fusionMethod);
        Map<String, Object> result = new HashMap<>();
        result.put("algorithm", fusionMethod);
        result.put("count", count);
        result.put("nextFileName", String.format("%s_%03d.jpg", fusionMethod, count + 1));
        return AjaxResult.success(result);
    }

    // 重置算法计数器
    @PostMapping("/reset-counter/{fusionMethod}")
    public AjaxResult resetAlgorithmCounter(@PathVariable String fusionMethod) {
        fusionServiceImpl.resetCounter(fusionMethod);
        return AjaxResult.success("算法 " + fusionMethod + " 的计数器已重置");
    }

    // 获取所有算法的计数统计
    @GetMapping("/count-stats")
    public AjaxResult getAllCountStats() {
        List<String> algorithms = Arrays.asList("fast", "wavelet", "pyramid", "sparse", "cnn", "self-encoder", "gan", "ganresnet");
        List<Map<String, Object>> stats = new ArrayList<>();
        
        for (String algorithm : algorithms) {
            int count = fusionServiceImpl.getCurrentCount(algorithm);
            Map<String, Object> stat = new HashMap<>();
            stat.put("algorithm", algorithm);
            stat.put("count", count);
            stat.put("nextFileName", String.format("%s_%03d.jpg", algorithm, count + 1));
            stats.add(stat);
        }
        
        return AjaxResult.success(stats);
    }

    /**
     * 根据融合方法获取参数列表
     */
    private List<Map<String, Object>> getParametersByMethod(String fusionMethod) {
        List<Map<String, Object>> params = new ArrayList<>();
        
        switch (fusionMethod) {
            case "fast":
                params.add(createParam("method", "融合方法", "最优寻找方法"));
                params.add(createParam("description", "算法描述", "快速融合算法，基于像素级的直接融合"));
                params.add(createParam("complexity", "算法复杂度", "O(n)"));
                params.add(createParam("performance", "性能特点", "速度快，适用于实时应用"));
                break;
            case "wavelet":
                params.add(createParam("method", "融合方法", "小波变换融合"));
                params.add(createParam("waveletType", "小波基函数", "db3"));
                params.add(createParam("decompositionLevel", "分解层数", "2"));
                params.add(createParam("lowFreqRule", "低频融合规则", "取平均值"));
                params.add(createParam("highFreqRule", "高频融合规则", "梯度加权融合"));
                params.add(createParam("enhancement", "增强处理", "非锐化掩膜增强"));
                params.add(createParam("blurKernel", "模糊核大小", "5x5"));
                break;
            case "pyramid":
                params.add(createParam("method", "融合方法", "拉普拉斯金字塔融合"));
                params.add(createParam("levels", "金字塔层数", "4"));
                params.add(createParam("pyramidType", "金字塔类型", "高斯+拉普拉斯"));
                params.add(createParam("fusionRule", "融合规则", "基于方差的高频加权"));
                params.add(createParam("varianceWindow", "方差计算窗口", "3x3"));
                params.add(createParam("edgeEnhancement", "边缘增强", "Canny边缘检测"));
                params.add(createParam("cannyThreshold", "Canny阈值", "50-150"));
                params.add(createParam("edgeWeight", "边缘权重", "0.3"));
                break;
            case "sparse":
                params.add(createParam("method", "融合方法", "稀疏表示融合"));
                params.add(createParam("dependency", "依赖库", "scikit-learn"));
                params.add(createParam("dictionarySize", "字典大小", "256"));
                params.add(createParam("patchSize", "分块大小", "8x8"));
                params.add(createParam("sparsityLevel", "稀疏度", "0.1"));
                break;
            case "cnn":
                params.add(createParam("method", "融合方法", "卷积神经网络融合"));
                params.add(createParam("dependency", "依赖库", "PyTorch"));
                params.add(createParam("modelFile", "模型文件", "cnn_fusion_model.pth"));
                params.add(createParam("architecture", "网络架构", "深度卷积网络"));
                params.add(createParam("inputChannels", "输入通道", "6 (可见光3+红外3)"));
                params.add(createParam("outputChannels", "输出通道", "3 (RGB)"));
                break;
            case "self-encoder":
                params.add(createParam("method", "融合方法", "自编码器融合"));
                params.add(createParam("dependency", "依赖库", "PyTorch"));
                params.add(createParam("architecture", "网络架构", "编码器-解码器"));
                params.add(createParam("encoderLayers", "编码器层数", "多层卷积编码"));
                params.add(createParam("decoderLayers", "解码器层数", "多层卷积解码"));
                params.add(createParam("latentSpace", "潜在空间", "压缩特征表示"));
                break;
            case "gan":
                params.add(createParam("method", "融合方法", "生成对抗网络融合"));
                params.add(createParam("dependency", "依赖库", "PyTorch"));
                params.add(createParam("generator", "生成器", "卷积生成网络"));
                params.add(createParam("discriminator", "判别器", "卷积判别网络"));
                params.add(createParam("lossFunction", "损失函数", "对抗损失+内容损失"));
                params.add(createParam("trainingMode", "训练模式", "对抗训练"));
                break;
            case "ganresnet":
                params.add(createParam("method", "融合方法", "GAN-ResNet融合"));
                params.add(createParam("dependency", "依赖库", "PyTorch"));
                params.add(createParam("generator", "生成器", "ResNet + GAN"));
                params.add(createParam("resnetBlocks", "ResNet块", "残差连接块"));
                params.add(createParam("skipConnections", "跳跃连接", "ResNet残差连接"));
                params.add(createParam("discriminator", "判别器", "卷积判别网络"));
                params.add(createParam("architecture", "网络架构", "ResNet + 对抗网络"));
                break;
            default:
                params.add(createParam("method", "融合方法", fusionMethod));
                params.add(createParam("status", "状态", "未实现的算法"));
                break;
        }
        
        return params;
    }

    /**
     * 创建参数对象
     */
    private Map<String, Object> createParam(String key, String label, String value) {
        Map<String, Object> param = new HashMap<>();
        param.put("key", key);
        param.put("label", label);
        param.put("value", value);
        return param;
    }
}

class FusionRequest {
    private String visibleImage;
    private String infraredImage;
    private String fusionMethod; // 新增融合方法参数："wavelet" 或 "pyramid"
    
    // getters and setters
    public String getVisibleImage() {
        return visibleImage;
    }
    
    public void setVisibleImage(String visibleImage) {
        this.visibleImage = visibleImage;
    }
    
    public String getInfraredImage() {
        return infraredImage;
    }
    
    public void setInfraredImage(String infraredImage) {
        this.infraredImage = infraredImage;
    }

    public String getFusionMethod() {
        return fusionMethod;
    }

    public void setFusionMethod(String fusionMethod) {
        this.fusionMethod = fusionMethod;
    }
}