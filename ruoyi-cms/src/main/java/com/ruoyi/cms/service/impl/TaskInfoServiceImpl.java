package com.ruoyi.cms.service.impl;

import com.ruoyi.cms.domain.TaskInfo;
import com.ruoyi.cms.service.TaskInfoService;
import org.mybatis.logging.Logger;
import org.mybatis.logging.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class TaskInfoServiceImpl implements TaskInfoService {
    private static final Logger log = LoggerFactory.getLogger(ImageFusionServiceImpl.class);

    private static final String TASK_INFO_KEY_PREFIX = "task_info:";
    private static final long TASK_INFO_EXPIRE_TIME = 3600; // 1小时过期

    @Autowired
    private RedisTemplate<Object, Object> redisTemplate;

    @Override
    public void saveTask(TaskInfo taskInfo) {
        String key = TASK_INFO_KEY_PREFIX + taskInfo.getTaskId();
        redisTemplate.opsForValue().set(key, taskInfo);
    }

    @Override
    public TaskInfo getTask(String taskId) {
        String key = TASK_INFO_KEY_PREFIX + taskId;
        return (TaskInfo) redisTemplate.opsForValue().get(key);
    }

    @Override
    public void updateTask(TaskInfo taskInfo) {
        saveTask(taskInfo);
    }
}