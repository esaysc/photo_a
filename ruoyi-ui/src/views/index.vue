<template>
  <div class="app-container home">
    <el-row :gutter="20">
      <el-col :sm="24" :lg="12" style="padding-left: 20px">
        <h2>图像融合框架</h2>
        <p>
          <b>当前版本:</b> <span>v{{ version }}</span>
        </p>
        <p class="system-intro">
          本系统是一个基于深度学习的图像融合框架，支持多种融合算法，包括CNN、GAN、自编码器等方法。系统提供了友好的Web界面，方便用户进行图像融合实验和结果分析。
        </p>
        <h3>主要功能</h3>
        <ul class="feature-list">
          <li><el-icon><Picture /></el-icon> 支持多种图像格式的上传和预览</li>
          <li><el-icon><Operation /></el-icon> 提供多种先进的融合算法</li>
          <li><el-icon><Monitor /></el-icon> 实时展示融合过程和结果</li>
          <li><el-icon><DataAnalysis /></el-icon> 融合效果量化评估</li>
        </ul>
      </el-col>

      <el-col :sm="24" :lg="12" style="padding-left: 50px">
        <el-row>
          <el-col :span="12">
            <h2>技术选型</h2>
          </el-col>
        </el-row>
        <el-row>
          <el-col :span="6">
            <h4>后端技术</h4>
            <ul>
              <li>SpringBoot</li>
              <li>Spring Security</li>
              <li>JWT</li>
              <li>MyBatis</li>
              <li>Druid</li>
              <li>Fastjson</li>
              <li>Python</li>
              <li>PyTorch</li>
            </ul>
          </el-col>
          <el-col :span="6">
            <h4>前端技术</h4>
            <ul>
              <li>Vue</li>
              <li>Vuex</li>
              <li>Element Plus</li>
              <li>Axios</li>
              <li>Sass</li>
              <li>ECharts</li>
              <li>OpenCV.js</li>
            </ul>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
    <el-divider />
    <el-row :gutter="20">
      <el-col :xs="24" :sm="24" :md="12" :lg="8">
        <el-card class="feature-card">
          <template #header>
            <div class="card-header">
              <h3><el-icon><Guide /></el-icon> 使用指南</h3>
            </div>
          </template>
          <div class="card-content">
            <ol>
              <li>在图像管理中上传待融合的图像</li>
              <li>选择合适的融合算法和参数</li>
              <li>启动融合任务并等待结果</li>
              <li>下载或分析融合结果</li>
            </ol>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="24" :md="12" :lg="8">
        <el-card class="feature-card">
          <template #header>
            <div class="card-header">
              <h3><el-icon><Cpu /></el-icon> 算法特点</h3>
            </div>
          </template>
          <div class="card-content">
            <ul>
              <li>CNN：适用于一般场景的图像融合</li>
              <li>GAN：生成更自然的融合效果</li>
              <li>自编码器：保持图像细节信息</li>
              <li>稀疏表示：适合特定场景融合</li>
            </ul>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="24" :md="12" :lg="8">
        <el-card class="feature-card">
          <template #header>
            <div class="card-header">
              <h3><el-icon><Warning /></el-icon> 注意事项</h3>
            </div>
          </template>
          <div class="card-content">
            <ul>
              <li>支持的图像格式：PNG、JPG、BMP</li>
              <li>建议图像分辨率：小于2048×2048</li>
              <li>图像大小限制：小于10MB</li>
              <li>处理时间：根据算法不同约1-5分钟</li>
            </ul>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup name="Index">
import { ref } from 'vue'
import { Picture, Operation, Monitor, DataAnalysis, Guide, Cpu, Warning } from '@element-plus/icons-vue'

const version = ref('3.8.9')

function goTarget(url) {
  window.open(url, '__blank')
}
</script>

<style scoped lang="scss">
.home {
  blockquote {
    padding: 10px 20px;
    margin: 0 0 20px;
    font-size: 17.5px;
    border-left: 5px solid #eee;
  }
  hr {
    margin-top: 20px;
    margin-bottom: 20px;
    border: 0;
    border-top: 1px solid #eee;
  }
  .col-item {
    margin-bottom: 20px;
  }

  font-family: "open sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 14px;
  color: #676a6c;
  overflow-x: hidden;

  ul {
    padding: 0;
    margin: 0;
    list-style-type: none;
  }

  h4 {
    margin-top: 0px;
  }

  h2 {
    margin-top: 10px;
    font-size: 26px;
    font-weight: 100;
  }

  h3 {
    margin-top: 20px;
    font-size: 18px;
    font-weight: 400;
  }

  p {
    margin-top: 10px;
    line-height: 1.6;

    b {
      font-weight: 700;
    }
  }

  .system-intro {
    margin: 20px 0;
    font-size: 15px;
    line-height: 1.8;
    color: #555;
  }

  .feature-list {
    margin: 20px 0;
    li {
      margin: 10px 0;
      display: flex;
      align-items: center;
      .el-icon {
        margin-right: 8px;
        color: #409EFF;
      }
    }
  }

  .feature-card {
    margin-bottom: 20px;
    .card-header {
      display: flex;
      align-items: center;
      h3 {
        margin: 0;
        .el-icon {
          margin-right: 8px;
          color: #409EFF;
        }
      }
    }
    .card-content {
      padding: 10px 0;
      li {
        margin: 8px 0;
        line-height: 1.6;
      }
    }
  }
}
</style>

