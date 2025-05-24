package com.ruoyi.cms.mapper;

import java.util.List;
import com.ruoyi.cms.domain.VideoResource;

/**
 * 视频Mapper接口
 * 
 * @author lhl
 * @date 2025-05-22
 */
public interface VideoResourceMapper 
{
    /**
     * 查询视频
     * 
     * @param id 视频主键
     * @return 视频
     */
    public VideoResource selectVideoResourceById(Long id);

    /**
     * 查询视频列表
     * 
     * @param videoResource 视频
     * @return 视频集合
     */
    public List<VideoResource> selectVideoResourceList(VideoResource videoResource);

    /**
     * 新增视频
     * 
     * @param videoResource 视频
     * @return 结果
     */
    public int insertVideoResource(VideoResource videoResource);

    /**
     * 修改视频
     * 
     * @param videoResource 视频
     * @return 结果
     */
    public int updateVideoResource(VideoResource videoResource);

    /**
     * 删除视频
     * 
     * @param id 视频主键
     * @return 结果
     */
    public int deleteVideoResourceById(Long id);

    /**
     * 批量删除视频
     * 
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteVideoResourceByIds(Long[] ids);
}
