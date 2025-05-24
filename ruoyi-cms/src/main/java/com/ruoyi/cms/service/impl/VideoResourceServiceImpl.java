package com.ruoyi.cms.service.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.cms.mapper.VideoResourceMapper;
import com.ruoyi.cms.domain.VideoResource;
import com.ruoyi.cms.service.IVideoResourceService;

/**
 * 视频Service业务层处理
 * 
 * @author lhl
 * @date 2025-05-22
 */
@Service
public class VideoResourceServiceImpl implements IVideoResourceService 
{
    @Autowired
    private VideoResourceMapper videoResourceMapper;

    /**
     * 查询视频
     * 
     * @param id 视频主键
     * @return 视频
     */
    @Override
    public VideoResource selectVideoResourceById(Long id)
    {
        return videoResourceMapper.selectVideoResourceById(id);
    }

    /**
     * 查询视频列表
     * 
     * @param videoResource 视频
     * @return 视频
     */
    @Override
    public List<VideoResource> selectVideoResourceList(VideoResource videoResource)
    {
        return videoResourceMapper.selectVideoResourceList(videoResource);
    }

    /**
     * 新增视频
     * 
     * @param videoResource 视频
     * @return 结果
     */
    @Override
    public int insertVideoResource(VideoResource videoResource)
    {
        return videoResourceMapper.insertVideoResource(videoResource);
    }

    /**
     * 修改视频
     * 
     * @param videoResource 视频
     * @return 结果
     */
    @Override
    public int updateVideoResource(VideoResource videoResource)
    {
        return videoResourceMapper.updateVideoResource(videoResource);
    }

    /**
     * 批量删除视频
     * 
     * @param ids 需要删除的视频主键
     * @return 结果
     */
    @Override
    public int deleteVideoResourceByIds(Long[] ids)
    {
        return videoResourceMapper.deleteVideoResourceByIds(ids);
    }

    /**
     * 删除视频信息
     * 
     * @param id 视频主键
     * @return 结果
     */
    @Override
    public int deleteVideoResourceById(Long id)
    {
        return videoResourceMapper.deleteVideoResourceById(id);
    }
}
