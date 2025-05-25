package com.ruoyi.cms.mapper;

import java.util.List;
import com.ruoyi.cms.domain.BookResource;

/**
 * 图书Mapper接口
 * 
 * @author lhl
 * @date 2025-05-25
 */
public interface BookResourceMapper 
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
     * 删除图书
     * 
     * @param id 图书主键
     * @return 结果
     */
    public int deleteBookResourceById(Long id);

    /**
     * 批量删除图书
     * 
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteBookResourceByIds(Long[] ids);
}
