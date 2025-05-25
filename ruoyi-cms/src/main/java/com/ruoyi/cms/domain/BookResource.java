package com.ruoyi.cms.domain;

import java.util.Date;
import com.fasterxml.jackson.annotation.JsonFormat;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import com.ruoyi.common.annotation.Excel;
import com.ruoyi.common.core.domain.BaseEntity;

/**
 * 图书对象 book_resource
 * 
 * @author lhl
 * @date 2025-05-25
 */
public class BookResource extends BaseEntity
{
    private static final long serialVersionUID = 1L;

    /**  */
    private Long id;

    /** 书籍名称 */
    @Excel(name = "书籍名称")
    private String name;

    /** 文件存储路径 */
    @Excel(name = "文件存储路径")
    private String storagePath;

    /** 文件类型，如 PDF、EPUB */
    @Excel(name = "文件类型，如 PDF、EPUB")
    private String fileType;

    /** 书籍简介 */
    @Excel(name = "书籍简介")
    private String description;

    /** 适用人群 */
    @Excel(name = "适用人群")
    private String audience;

    /** 封面图像路径 */
    @Excel(name = "封面图像路径")
    private String coverPath;

    /** 关联 book_category */
    @Excel(name = "关联 book_category")
    private Long categoryId;

    /**  */
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

    public void setFileType(String fileType) 
    {
        this.fileType = fileType;
    }

    public String getFileType() 
    {
        return fileType;
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
            .append("fileType", getFileType())
            .append("description", getDescription())
            .append("audience", getAudience())
            .append("coverPath", getCoverPath())
            .append("categoryId", getCategoryId())
            .append("createdAt", getCreatedAt())
            .toString();
    }
}
