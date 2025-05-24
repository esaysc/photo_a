package com.ruoyi.cms.service;

import com.ruoyi.cms.domain.TaskInfo;

public interface TaskInfoService {
    void saveTask(TaskInfo taskInfo);
    void updateTask(TaskInfo taskInfo);
    TaskInfo getTask(String taskId);
}