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
import com.ruoyi.cms.domain.VideoResource;
import com.ruoyi.cms.service.IVideoResourceService;
import com.ruoyi.common.utils.poi.ExcelUtil;
import com.ruoyi.common.core.page.TableDataInfo;

/**
 * 视频Controller
 * 
 * @author lhl
 * @date 2025-05-22
 */
@RestController
@RequestMapping("/cms/video")
public class VideoResourceController extends BaseController
{
    @Autowired
    private IVideoResourceService videoResourceService;

    /**
     * 查询视频列表
     */
    @PreAuthorize("@ss.hasPermi('cms:video:list')")
    @GetMapping("/list")
    public TableDataInfo list(VideoResource videoResource)
    {
        startPage();
        List<VideoResource> list = videoResourceService.selectVideoResourceList(videoResource);
        return getDataTable(list);
    }

    /**
     * 导出视频列表
     */
    @PreAuthorize("@ss.hasPermi('cms:video:export')")
    @Log(title = "视频", businessType = BusinessType.EXPORT)
    @PostMapping("/export")
    public void export(HttpServletResponse response, VideoResource videoResource)
    {
        List<VideoResource> list = videoResourceService.selectVideoResourceList(videoResource);
        ExcelUtil<VideoResource> util = new ExcelUtil<VideoResource>(VideoResource.class);
        util.exportExcel(response, list, "视频数据");
    }

    /**
     * 获取视频详细信息
     */
    @PreAuthorize("@ss.hasPermi('cms:video:query')")
    @GetMapping(value = "/{id}")
    public AjaxResult getInfo(@PathVariable("id") Long id)
    {
        return success(videoResourceService.selectVideoResourceById(id));
    }

    /**
     * 新增视频
     */
    @PreAuthorize("@ss.hasPermi('cms:video:add')")
    @Log(title = "视频", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@RequestBody VideoResource videoResource)
    {
        return toAjax(videoResourceService.insertVideoResource(videoResource));
    }

    /**
     * 修改视频
     */
    @PreAuthorize("@ss.hasPermi('cms:video:edit')")
    @Log(title = "视频", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@RequestBody VideoResource videoResource)
    {
        return toAjax(videoResourceService.updateVideoResource(videoResource));
    }

    /**
     * 删除视频
     */
    @PreAuthorize("@ss.hasPermi('cms:video:remove')")
    @Log(title = "视频", businessType = BusinessType.DELETE)
	@DeleteMapping("/{ids}")
    public AjaxResult remove(@PathVariable Long[] ids)
    {
        return toAjax(videoResourceService.deleteVideoResourceByIds(ids));
    }
}
