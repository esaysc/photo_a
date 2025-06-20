<template>
  <div class="app-container">
    <el-card class="manual-card">
      <template #header>
        <div class="card-header">
          <h2><el-icon><Document /></el-icon> 系统使用说明</h2>
        </div>
      </template>
      <div class="manual-content" v-html="compiledContent"></div>
    </el-card>
  </div>
</template>

<script setup name="Info">
import { ref, onMounted } from 'vue'
import { Document } from '@element-plus/icons-vue'
import { marked } from 'marked'

const manualContent = `
# 图像融合系统使用说明

## 1. 系统概述
本系统是一个基于深度学习的图像融合框架，支持多种融合算法，包括 CNN、GAN、自编码器等方法。系统提供了友好的 Web 界面，方便用户进行图像融合实验和结果分析。

## 2. 功能说明
- **图像上传与管理**：支持多种格式图像的上传、预览和管理
- **算法选择**：提供多种先进的融合算法供选择
- **参数配置**：支持算法参数的自定义配置
- **结果分析**：提供融合结果的定量评估和可视化分析
- **批量处理**：支持多组图像的批量融合处理

## 3. 使用流程
1. **准备数据**
   - 准备待融合的图像文件（支持 PNG、JPG、BMP 格式）
   - 确保图像分辨率合适（建议不超过 2048×2048）
   - 图像大小应小于 10MB

2. **上传图像**
   - 进入图像管理页面
   - 点击上传按钮选择文件
   - 等待上传完成

3. **选择算法**
   - 进入融合任务页面
   - 选择需要融合的图像对
   - 选择合适的融合算法
   - 配置算法参数（可选）

4. **执行融合**
   - 点击开始融合按钮
   - 等待融合完成（约 1-5 分钟）
   - 查看融合结果

5. **结果分析**
   - 查看融合图像
   - 分析评估指标
   - 下载结果

## 4. 注意事项
- 请确保上传图像质量良好，无明显噪声和失真
- 融合过程中请勿刷新页面或关闭浏览器
- 建议使用 Chrome 或 Firefox 最新版本浏览器
- 系统支持断点续传，但建议保持网络稳定

## 5. 算法说明

### 5.1 CNN 融合
- 适用场景：通用图像融合
- 特点：保持图像结构信息
- 处理时间：约 1-2 分钟

### 5.2 GAN 融合
- 适用场景：需要高质量视觉效果
- 特点：生成更自然的融合结果
- 处理时间：约 2-3 分钟

### 5.3 自编码器融合
- 适用场景：需要保持细节信息
- 特点：良好的特征提取能力
- 处理时间：约 1-2 分钟

### 5.4 稀疏表示融合
- 适用场景：特定场景的精确融合
- 特点：适合处理特定类型图像
- 处理时间：约 3-5 分钟

## 6. 常见问题

### 6.1 上传失败
- 检查文件格式是否正确
- 确认文件大小是否超限
- 检查网络连接是否正常

### 6.2 融合时间过长
- 检查图像大小是否合适
- 确认服务器负载情况
- 考虑使用更轻量级的算法

### 6.3 结果不理想
- 检查输入图像质量
- 调整算法参数
- 尝试其他融合算法

### 6.4 系统异常
- 刷新页面重试
- 清除浏览器缓存
- 联系技术支持
`

const compiledContent = ref('')

onMounted(() => {
  compiledContent.value = marked(manualContent)
})
</script>

<style scoped lang="scss">
.manual-card {
  margin: 20px;
  
  .card-header {
    display: flex;
    align-items: center;
    
    h2 {
      margin: 0;
      font-size: 20px;
      font-weight: 500;
      
      .el-icon {
        margin-right: 8px;
        color: #409EFF;
      }
    }
  }
}

.manual-content {
  padding: 20px;
  font-size: 15px;
  line-height: 1.8;
  color: #333;
  
  :deep(h1) {
    font-size: 28px;
    font-weight: 500;
    margin: 20px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
  }
  
  :deep(h2) {
    font-size: 24px;
    font-weight: 500;
    margin: 20px 0 15px;
    padding-bottom: 8px;
    border-bottom: 1px solid #eee;
  }
  
  :deep(h3) {
    font-size: 20px;
    font-weight: 500;
    margin: 15px 0 10px;
  }
  
  :deep(p) {
    margin: 10px 0;
    line-height: 1.8;
  }
  
  :deep(ul), :deep(ol) {
    padding-left: 20px;
    margin: 10px 0;
    
    li {
      margin: 5px 0;
    }
  }
  
  :deep(code) {
    background-color: #f5f5f5;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: Consolas, Monaco, 'Courier New', monospace;
  }
  
  :deep(strong) {
    font-weight: 600;
    color: #000;
  }
}
</style>