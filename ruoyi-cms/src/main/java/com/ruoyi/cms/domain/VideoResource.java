package com.ruoyi.cms.domain;

import java.util.Date;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * 视频对象 video_resource
 * 
 * @author lhl
 * @date 2025-05-22
 */
public class VideoResource extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /** $column.columnComment */
    private Long id;

    /** 视频标题 */
    @Excel(name = "视频标题")
    private String name;

    /** 文件存储路径 */
    @Excel(name = "文件存储路径")
    private String storagePath;

    /** 视频时长 */
    @JsonFormat(pattern = "yyyy-MM-dd")
    @Excel(name = "视频时长", width = 30, dateFormat = "yyyy-MM-dd")
    private Date duration;

    /** 视频简介 */
    @Excel(name = "视频简介")
    private String description;

    /** 适用人群 */
    @Excel(name = "适用人群")
    private String audience;

    /** 封面图像路径 */
    @Excel(name = "封面图像路径")
    private String coverPath;

    /** 关联 video_category */
    @Excel(name = "关联 video_category")
    private Long categoryId;

    /** $column.columnComment */
    private Date createdAt;

    public void setId(Long id) 
    {
        this.id = id;
    }

    public Long getId() 
    {
        return id;
    }

    public void setName(String name) 
    {
        this.name = name;
    }

    public String getName() 
    {
        return name;
    }

    public void setStoragePath(String storagePath) 
    {
        this.storagePath = storagePath;
    }

    public String getStoragePath() 
    {
        return storagePath;
    }

    public void setDuration(Date duration) 
    {
        this.duration = duration;
    }

    public Date getDuration() 
    {
        return duration;
    }

    public void setDescription(String description) 
    {
        this.description = description;
    }

    public String getDescription() 
    {
        return description;
    }

    public void setAudience(String audience) 
    {
        this.audience = audience;
    }

    public String getAudience() 
    {
        return audience;
    }

    public void setCoverPath(String coverPath) 
    {
        this.coverPath = coverPath;
    }

    public String getCoverPath() 
    {
        return coverPath;
    }

    public void setCategoryId(Long categoryId) 
    {
        this.categoryId = categoryId;
    }

    public Long getCategoryId() 
    {
        return categoryId;
    }

    public void setCreatedAt(Date createdAt) 
    {
        this.createdAt = createdAt;
    }

    public Date getCreatedAt() 
    {
        return createdAt;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this,ToStringStyle.MULTI_LINE_STYLE)
            .append("id", getId())
            .append("name", getName())
            .append("storagePath", getStoragePath())
            .append("duration", getDuration())
            .append("description", getDescription())
            .append("audience", getAudience())
            .append("coverPath", getCoverPath())
            .append("categoryId", getCategoryId())
            .append("createdAt", getCreatedAt())
            .toString();
    }
}
