package com.ruoyi.cms.service.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.cms.mapper.BookCategoryMapper;
import com.ruoyi.cms.domain.BookCategory;
import com.ruoyi.cms.service.IBookCategoryService;

/**
 * 图书分类Service业务层处理
 * 
 * @author lhl
 * @date 2025-05-21
 */
@Service
public class BookCategoryServiceImpl implements IBookCategoryService 
{
    @Autowired
    private BookCategoryMapper bookCategoryMapper;

    /**
     * 查询图书分类
     * 
     * @param id 图书分类主键
     * @return 图书分类
     */
    @Override
    public BookCategory selectBookCategoryById(Long id)
    {
        return bookCategoryMapper.selectBookCategoryById(id);
    }

    /**
     * 查询图书分类列表
     * 
     * @param bookCategory 图书分类
     * @return 图书分类
     */
    @Override
    public List<BookCategory> selectBookCategoryList(BookCategory bookCategory)
    {
        return bookCategoryMapper.selectBookCategoryList(bookCategory);
    }

    /**
     * 新增图书分类
     * 
     * @param bookCategory 图书分类
     * @return 结果
     */
    @Override
    public int insertBookCategory(BookCategory bookCategory)
    {
        return bookCategoryMapper.insertBookCategory(bookCategory);
    }

    /**
     * 修改图书分类
     * 
     * @param bookCategory 图书分类
     * @return 结果
     */
    @Override
    public int updateBookCategory(BookCategory bookCategory)
    {
        return bookCategoryMapper.updateBookCategory(bookCategory);
    }

    /**
     * 批量删除图书分类
     * 
     * @param ids 需要删除的图书分类主键
     * @return 结果
     */
    @Override
    public int deleteBookCategoryByIds(Long[] ids)
    {
        return bookCategoryMapper.deleteBookCategoryByIds(ids);
    }

    /**
     * 删除图书分类信息
     * 
     * @param id 图书分类主键
     * @return 结果
     */
    @Override
    public int deleteBookCategoryById(Long id)
    {
        return bookCategoryMapper.deleteBookCategoryById(id);
    }
}
