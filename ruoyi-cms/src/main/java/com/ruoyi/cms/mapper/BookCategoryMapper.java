package com.ruoyi.cms.mapper;

import java.util.List;
import com.ruoyi.cms.domain.BookCategory;

/**
 * 图书分类Mapper接口
 * 
 * @author lhl
 * @date 2025-05-21
 */
public interface BookCategoryMapper 
{
    /**
     * 查询图书分类
     * 
     * @param id 图书分类主键
     * @return 图书分类
     */
    public BookCategory selectBookCategoryById(Long id);

    /**
     * 查询图书分类列表
     * 
     * @param bookCategory 图书分类
     * @return 图书分类集合
     */
    public List<BookCategory> selectBookCategoryList(BookCategory bookCategory);

    /**
     * 新增图书分类
     * 
     * @param bookCategory 图书分类
     * @return 结果
     */
    public int insertBookCategory(BookCategory bookCategory);

    /**
     * 修改图书分类
     * 
     * @param bookCategory 图书分类
     * @return 结果
     */
    public int updateBookCategory(BookCategory bookCategory);

    /**
     * 删除图书分类
     * 
     * @param id 图书分类主键
     * @return 结果
     */
    public int deleteBookCategoryById(Long id);

    /**
     * 批量删除图书分类
     * 
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteBookCategoryByIds(Long[] ids);
}
