package com.ruoyi.cms.controller;

import java.util.List;
import javax.servlet.http.HttpServletResponse;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.ruoyi.common.annotation.Log;
import com.ruoyi.common.core.controller.BaseController;
import com.ruoyi.common.core.domain.AjaxResult;
import com.ruoyi.common.enums.BusinessType;
import com.ruoyi.cms.domain.BookResource;
import com.ruoyi.cms.service.IBookResourceService;
import com.ruoyi.common.utils.poi.ExcelUtil;
import com.ruoyi.common.core.page.TableDataInfo;

/**
 * 图书Controller
 * 
 * @author lhl
 * @date 2025-05-22
 */
@RestController
@RequestMapping("/cms/book")
public class BookResourceController extends BaseController
{
    @Autowired
    private IBookResourceService bookResourceService;

    /**
     * 查询图书列表
     */
    @PreAuthorize("@ss.hasPermi('cms:book:list')")
    @GetMapping("/list")
    public TableDataInfo list(BookResource bookResource)
    {
        startPage();
        List<BookResource> list = bookResourceService.selectBookResourceList(bookResource);
        return getDataTable(list);
    }

    /**
     * 导出图书列表
     */
    @PreAuthorize("@ss.hasPermi('cms:book:export')")
    @Log(title = "图书", businessType = BusinessType.EXPORT)
    @PostMapping("/export")
    public void export(HttpServletResponse response, BookResource bookResource)
    {
        List<BookResource> list = bookResourceService.selectBookResourceList(bookResource);
        ExcelUtil<BookResource> util = new ExcelUtil<BookResource>(BookResource.class);
        util.exportExcel(response, list, "图书数据");
    }

    /**
     * 获取图书详细信息
     */
    @PreAuthorize("@ss.hasPermi('cms:book:query')")
    @GetMapping(value = "/{id}")
    public AjaxResult getInfo(@PathVariable("id") Long id)
    {
        return success(bookResourceService.selectBookResourceById(id));
    }

    /**
     * 新增图书
     */
    @PreAuthorize("@ss.hasPermi('cms:book:add')")
    @Log(title = "图书", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@RequestBody BookResource bookResource)
    {
        return toAjax(bookResourceService.insertBookResource(bookResource));
    }

    /**
     * 修改图书
     */
    @PreAuthorize("@ss.hasPermi('cms:book:edit')")
    @Log(title = "图书", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@RequestBody BookResource bookResource)
    {
        return toAjax(bookResourceService.updateBookResource(bookResource));
    }

    /**
     * 删除图书
     */
    @PreAuthorize("@ss.hasPermi('cms:book:remove')")
    @Log(title = "图书", businessType = BusinessType.DELETE)
	@DeleteMapping("/{ids}")
    public AjaxResult remove(@PathVariable Long[] ids)
    {
        return toAjax(bookResourceService.deleteBookResourceByIds(ids));
    }
}
