package com.ruoyi.cms.service.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.cms.mapper.BookResourceMapper;
import com.ruoyi.cms.domain.BookResource;
import com.ruoyi.cms.service.IBookResourceService;

/**
 * 图书Service业务层处理
 * 
 * @author lhl
 * @date 2025-05-25
 */
@Service
public class BookResourceServiceImpl implements IBookResourceService 
{
    @Autowired
    private BookResourceMapper bookResourceMapper;

    /**
     * 查询图书
     * 
     * @param id 图书主键
     * @return 图书
     */
    @Override
    public BookResource selectBookResourceById(Long id)
    {
        return bookResourceMapper.selectBookResourceById(id);
    }

    /**
     * 查询图书列表
     * 
     * @param bookResource 图书
     * @return 图书
     */
    @Override
    public List<BookResource> selectBookResourceList(BookResource bookResource)
    {
        return bookResourceMapper.selectBookResourceList(bookResource);
    }

    /**
     * 新增图书
     * 
     * @param bookResource 图书
     * @return 结果
     */
    @Override
    public int insertBookResource(BookResource bookResource)
    {
        return bookResourceMapper.insertBookResource(bookResource);
    }

    /**
     * 修改图书
     * 
     * @param bookResource 图书
     * @return 结果
     */
    @Override
    public int updateBookResource(BookResource bookResource)
    {
        return bookResourceMapper.updateBookResource(bookResource);
    }

    /**
     * 批量删除图书
     * 
     * @param ids 需要删除的图书主键
     * @return 结果
     */
    @Override
    public int deleteBookResourceByIds(Long[] ids)
    {
        return bookResourceMapper.deleteBookResourceByIds(ids);
    }

    /**
     * 删除图书信息
     * 
     * @param id 图书主键
     * @return 结果
     */
    @Override
    public int deleteBookResourceById(Long id)
    {
        return bookResourceMapper.deleteBookResourceById(id);
    }
}
