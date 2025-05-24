package com.ruoyi.cms.controller;


import com.ruoyi.cms.domain.TaskInfo;
import com.ruoyi.cms.service.IImageFusionService;
import com.ruoyi.cms.service.TaskInfoService;
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

@RestController
@RequestMapping("/cms/fusion")
public class FusionController {

    @Autowired
    private IImageFusionService fusionService;

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