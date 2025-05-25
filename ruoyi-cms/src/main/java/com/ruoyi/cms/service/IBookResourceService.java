package com.ruoyi.cms.service;

import java.util.List;
import com.ruoyi.cms.domain.BookResource;

/**
 * 图书Service接口
 * 
 * @author lhl
 * @date 2025-05-25
 */
public interface IBookResourceService 
{
    /**
     * 查询图书
     * 
     * @param id 图书主键
     * @return 图书
     */
    public BookResource selectBookResourceById(Long id);

    /**
     * 查询图书列表
     * 
     * @param bookResource 图书
     * @return 图书集合
     */
    public List<BookResource> selectBookResourceList(BookResource bookResource);

    /**
     * 新增图书
     * 
     * @param bookResource 图书
     * @return 结果
     */
    public int insertBookResource(BookResource bookResource);

    /**
     * 修改图书
     * 
     * @param bookResource 图书
     * @return 结果
     */
    public int updateBookResource(BookResource bookResource);

    /**
     * 批量删除图书
     * 
     * @param ids 需要删除的图书主键集合
     * @return 结果
     */
    public int deleteBookResourceByIds(Long[] ids);

    /**
     * 删除图书信息
     * 
     * @param id 图书主键
     * @return 结果
     */
    public int deleteBookResourceById(Long id);
}
