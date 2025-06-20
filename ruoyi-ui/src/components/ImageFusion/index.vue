<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>{{ title }}</span>
      </div>
      
      <!-- 添加算法参数展示区域 -->
      <AlgorithmParams :fusion-method="fusionMethod" />
      
      <el-row :gutter="20">
        <el-col :span="11">
          <div class="image-container">
            <div class="upload-title">可见光图片</div>
            <el-upload
              class="image-uploader"
              :action="uploadUrl"
              :show-file-list="false"
              :on-success="handleVisibleSuccess"
              :on-error="handleUploadError"
              :before-upload="beforeImageUpload"
              :headers="headers">
              <img v-if="visibleImageUrl" :src="visibleImageUrl" class="uploaded-image">
              <i v-else class="el-icon-plus upload-icon"></i>
            </el-upload>
            <div class="upload-tip">请上传可见光图片</div>
          </div>
        </el-col>
        <el-col :span="11">
          <div class="image-container">
            <div class="upload-title">红外光图片</div>
            <el-upload
              class="image-uploader"
              :action="uploadUrl"
              :show-file-list="false"
              :on-success="handleInfraredSuccess"
              :on-error="handleUploadError"
              :before-upload="beforeImageUpload"
              :headers="headers">
              <img v-if="infraredImageUrl" :src="infraredImageUrl" class="uploaded-image">
              <i v-else class="el-icon-plus upload-icon"></i>
            </el-upload>
            <div class="upload-tip">请上传红外光图片</div>
          </div>
        </el-col>
      </el-row>

      <el-row style="margin-top: 20px; text-align: center;">
        <el-button type="primary" @click="handleFusion" :loading="loading" :disabled="!canFusion">
          开始融合
        </el-button>
        <el-button @click="handleReset">重置</el-button>
        
        <!-- 添加进度条和耗时显示 -->
        <div v-if="loading" style="margin-top: 20px;">
          <el-progress 
            :percentage="progress"
            :status="progress === 100 ? 'success' : ''"
            :stroke-width="20"
            :show-text="true"
            :format="progressFormat"
          />
          
          <!-- 优化后的实时耗时显示 -->
          <div class="fusion-status-panel">
            <div class="status-header">
              <div class="pulse-dot"></div>
              <span class="status-title">融合进行中</span>
            </div>
            
            <div class="status-content">
              <div class="primary-info">
                <div class="time-circle">
                  <div class="time-number">{{ elapsedTime }}</div>
                  <div class="time-unit">秒</div>
                </div>
                <div class="time-detail">
                  <div class="time-label">已耗时</div>
                  <div class="time-formatted">{{ formatTime(elapsedTime) }}</div>
                </div>
              </div>
              
              <div class="secondary-info">
                <div class="info-row">
                  <div class="info-item">
                    <el-tag 
                      :type="getStatusTagType()" 
                      effect="dark" 
                      size="medium"
                      class="status-tag"
                    >
                      <i :class="getStatusIcon()"></i>
                      {{ currentStatus }}
                    </el-tag>
                  </div>
                  <div class="info-item">
                    <el-tag 
                      type="info" 
                      effect="plain" 
                      size="medium"
                      class="algorithm-tag"
                    >
                      <i class="el-icon-cpu"></i>
                      {{ getAlgorithmDisplayName() }}
                    </el-tag>
                  </div>
                </div>
                
                <div class="progress-text">
                  进度: {{ progress }}% • 预计还需 {{ getEstimatedTime() }}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 优化后的完成显示 -->
        <div v-if="!loading && completedTime > 0" class="completion-panel">
          <div class="completion-content">
            <div class="completion-icon">
              <i class="el-icon-circle-check"></i>
            </div>
            <div class="completion-info">
              <div class="completion-title">融合完成！</div>
              <div class="completion-details">
                <span class="completion-time">耗时: {{ formatTime(completedTime) }}</span>
                <span class="completion-algorithm">算法: {{ getAlgorithmDisplayName() }}</span>
              </div>
            </div>
          </div>
        </div>
      </el-row>

      <el-row style="margin-top: 20px;" v-if="resultImageUrl">
        <el-col :span="24">
          <div class="result-container">
            <div class="result-title">融合结果</div>
            <img :src="resultImageUrl" class="result-image">
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script>
import { fusionImages } from "@/api/cms/fusion";
import { getToken } from '@/utils/auth'
import { getFusionProgress } from '@/api/cms/fusion'
import AlgorithmParams from '@/components/AlgorithmParams/index.vue'

export default {
  name: "ImageFusion",
  components: {
    AlgorithmParams
  },
  props: {
    // 页面标题
    title: {
      type: String,
      required: true
    },
    // 融合方法
    fusionMethod: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      uploadUrl: import.meta.env.VITE_APP_BASE_API + "/common/upload",
      visibleImageUrl: '',
      infraredImageUrl: '',
      resultImageUrl: '',
      loading: false,
      headers: {
        Authorization: 'Bearer ' + getToken()
      },
      taskId: null,           // 添加任务ID
      progress: 0,            // 添加进度值
      progressTimer: null,     // 添加进度查询定时器
      startTime: null,        // 添加开始时间
      elapsedTime: 0,         // 添加耗时（秒）
      timerInterval: null,    // 添加计时器
      completedTime: 0,       // 完成时的总耗时
      currentStatus: '准备中'  // 当前处理状态
    }
  },
  computed: {
    canFusion() {
      return this.visibleImageUrl && this.infraredImageUrl && !this.loading;
    }
  },
  methods: {
    // 格式化时间显示
    formatTime(seconds) {
      if (seconds < 60) {
        return `${seconds}秒`;
      } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}分${remainingSeconds}秒`;
      } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const remainingSeconds = seconds % 60;
        return `${hours}小时${minutes}分${remainingSeconds}秒`;
      }
    },

    // 更新处理状态
    updateStatus(progress) {
      if (progress === 0) {
        this.currentStatus = '初始化中';
      } else if (progress < 20) {
        this.currentStatus = '准备数据中';
      } else if (progress < 30) {
        this.currentStatus = '启动算法';
      } else if (progress < 80) {
        this.currentStatus = '正在融合';
      } else if (progress < 100) {
        this.currentStatus = '后处理中';
      } else {
        this.currentStatus = '完成';
      }
    },

    handleVisibleSuccess(response) {
      this.uploading = false;
      if (response.code === 200) {
        this.visibleImageUrl = response.url;
        this.$modal.msgSuccess('上传成功');
      } else {
        this.$modal.msgError(response.msg || '上传失败');
      }
    },
    handleInfraredSuccess(response) {
      this.uploading = false;
      if (response.code === 200) {
        this.infraredImageUrl = response.url;
        this.$modal.msgSuccess('上传成功');
      } else {
        this.$modal.msgError(response.msg || '上传失败');
      }
    },
    handleUploadError(err) {
      this.loading = false;
      const message = err.msg || '上传图片失败，请重试'
      this.$modal.msgError(message)
    },
    beforeImageUpload(file) {
      this.uploading = true;
      const isImage = file.type.indexOf('image/') === 0;
      const isLt2M = file.size / 1024 / 1024 < 2;

      if (!isImage) {
        this.$modal.msgError('只能上传图片文件!');
        this.uploading = false;
        return false;
      }
      if (!isLt2M) {
        this.$modal.msgError('图片大小不能超过 2MB!');
        this.uploading = false;
        return false;
      }
      return true;
    },
    handleFusion() {
      if (!this.canFusion) return;
      this.loading = true;
      this.progress = 0;
      this.completedTime = 0;
      this.currentStatus = '准备中';
      
      // 重置并启动计时器
      this.startTime = Date.now();
      this.elapsedTime = 0;
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
      }
      this.timerInterval = setInterval(() => {
        this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
      }, 1000);
      
      const getFilePath = (url) => {
        const baseUrl = 'http://localhost:8080';
        return url.replace(baseUrl, '').replace('/profile/', '');
      };
      
      fusionImages({
        visibleImage: getFilePath(this.visibleImageUrl),
        infraredImage: getFilePath(this.infraredImageUrl),
        fusionMethod: this.fusionMethod
      }).then(response => {
        if (response.code === 200) {
          this.taskId = response.data.taskId;  // 假设后端返回taskId
          this.startProgressPolling();         // 开始轮询进度
        }
      }).catch(() => {
        this.$modal.msgError("图像融合失败");
        this.loading = false;
        this.stopTimers();
      });
    },
    
    // 添加进度轮询方法
    startProgressPolling() {
      // 清除可能存在的旧定时器
      if (this.progressTimer) {
        clearInterval(this.progressTimer);
      }
      console.log("开始轮询");
      
      this.progressTimer = setInterval(async () => {
        try {
          const response = await getFusionProgress(this.taskId);
          console.log("respone => ", response);

          this.progress = response.data.progress;
          this.updateStatus(this.progress);
          
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            // 记录完成时的耗时
            this.completedTime = this.elapsedTime;
            
            // 停止所有计时器
            this.stopTimers();
            
            if (response.data.status === 'completed') {
              this.resultImageUrl = import.meta.env.VITE_APP_BASE_API + response.data.result;
              this.currentStatus = '完成';
              this.$modal.msgSuccess(`图像融合成功，耗时 ${this.formatTime(this.completedTime)}`);
            } else {
              this.$modal.msgError("图像融合失败");
            }
            
            this.loading = false;
          }
        } catch (error) {
          console.log("error => ", error)
          this.stopTimers();
          this.$modal.msgError("获取进度失败");
          this.loading = false;
        }
      }, 10000);
    },
    
    // 停止所有计时器
    stopTimers() {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
      if (this.progressTimer) {
        clearInterval(this.progressTimer);
        this.progressTimer = null;
      }
    },
    
    handleReset() {
      this.stopTimers();
      this.visibleImageUrl = '';
      this.infraredImageUrl = '';
      this.resultImageUrl = '';
      this.loading = false;
      this.progress = 0;
      this.taskId = null;
      this.startTime = null;
      this.elapsedTime = 0;
      this.completedTime = 0;
      this.currentStatus = '准备中';
    },
    
    beforeDestroy() {
      this.stopTimers();
    },

    // 新增方法
    progressFormat(percentage) {
      return `${percentage}%`;
    },
    
    getStatusTagType() {
      if (this.currentStatus === '完成') {
        return 'success';
      } else if (this.currentStatus === '失败') {
        return 'danger';
      } else if (this.currentStatus === '正在融合') {
        return 'warning';
      } else {
        return 'info';
      }
    },
    
    getStatusIcon() {
      if (this.currentStatus === '完成') {
        return 'el-icon-success';
      } else if (this.currentStatus === '失败') {
        return 'el-icon-error';
      } else if (this.currentStatus === '正在融合') {
        return 'el-icon-loading';
      } else {
        return 'el-icon-info';
      }
    },
    
    getAlgorithmDisplayName() {
      const algorithmNames = {
        'fast': '快速融合',
        'wavelet': '小波变换',
        'pyramid': '金字塔',
        'sparse': '稀疏表示',
        'cnn': '卷积神经网络',
        'self-encoder': '自编码器',
        'gan': '生成对抗网络',
        'ganresnet': 'GAN-ResNet'
      };
      return algorithmNames[this.fusionMethod] || this.fusionMethod.toUpperCase();
    },
    
    getEstimatedTime() {
      // 基于当前进度和已耗时计算预计剩余时间
      if (this.progress <= 0) {
        return '计算中...';
      }
      
      const estimatedTotal = (this.elapsedTime / this.progress) * 100;
      const remaining = Math.max(0, Math.ceil(estimatedTotal - this.elapsedTime));
      
      if (remaining <= 0) {
        return '即将完成';
      } else if (remaining < 60) {
        return `${remaining}秒`;
      } else {
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;
        return `${minutes}分${seconds}秒`;
      }
    }
  }
}
</script>

<style scoped>
.image-container {
  text-align: center;
  border: 1px solid #EBEEF5;
  border-radius: 4px;
  padding: 20px;
}

.upload-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 15px;
}

.image-uploader {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.image-uploader:hover {
  border-color: #409EFF;
}

.upload-icon {
  font-size: 28px;
  color: #8c939d;
  width: 300px;
  height: 300px;
  line-height: 300px;
  text-align: center;
}

.uploaded-image {
  width: 300px;
  height: 300px;
  display: block;
  object-fit: cover;
}

.upload-tip {
  font-size: 12px;
  color: #606266;
  margin-top: 10px;
}

.result-container {
  text-align: center;
  margin-top: 20px;
}

.result-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 15px;
}

.result-image {
  width: 600px;
  height: 400px;
  object-fit: contain;
  border: 1px solid #EBEEF5;
  border-radius: 4px;
  padding: 10px;
}

/* 新增耗时显示样式 */
.fusion-status-panel {
  background-color: #f5f7fa;
  border-radius: 8px;
  padding: 15px;
  margin-top: 15px;
}

.status-header {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.pulse-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #409EFF;
  margin-right: 5px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

.status-title {
  font-size: 14px;
  font-weight: bold;
  color: #303133;
}

.status-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
}

.primary-info {
  display: flex;
  align-items: center;
  flex: 1;
}

.time-circle {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #409EFF, #66B1FF);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin-right: 20px;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
  animation: rotate 8s linear infinite;
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.time-number {
  font-size: 20px;
  font-weight: bold;
  color: white;
}

.time-unit {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.8);
}

.time-detail {
  text-align: left;
}

.time-label {
  font-size: 14px;
  color: #606266;
  margin-bottom: 5px;
}

.time-formatted {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.secondary-info {
  flex: 1;
  text-align: right;
  min-width: 200px;
}

.info-row {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
  margin-bottom: 10px;
}

.info-item {
  display: flex;
  align-items: center;
}

.status-tag {
  margin-left: 5px;
}

.algorithm-tag {
  margin-left: 5px;
}

.progress-text {
  font-size: 12px;
  color: #606266;
  margin-top: 5px;
}

.completion-panel {
  margin-top: 15px;
  text-align: center;
}

.completion-content {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: linear-gradient(135deg, #f0fff4, #e6ffed);
  border-radius: 8px;
  border: 2px solid #52c41a;
  box-shadow: 0 4px 12px rgba(82, 196, 26, 0.2);
}

.completion-icon {
  font-size: 32px;
  color: #52c41a;
  margin-right: 15px;
  animation: bounce 1s ease-in-out;
}

@keyframes bounce {
  0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-10px); }
  60% { transform: translateY(-5px); }
}

.completion-info {
  text-align: left;
}

.completion-title {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 8px;
}

.completion-details {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.completion-time {
  font-size: 14px;
  color: #52c41a;
  font-weight: 600;
}

.completion-algorithm {
  font-size: 14px;
  color: #606266;
}
</style>