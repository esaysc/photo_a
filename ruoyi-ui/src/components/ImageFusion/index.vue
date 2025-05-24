<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>{{ title }}</span>
      </div>
      
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
        
        <!-- 添加进度条 -->
        <el-progress 
          v-if="loading" 
          style="margin-top: 20px;"
          :percentage="progress"
          :status="progress === 100 ? 'success' : ''"
          :stroke-width="15"
        />
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
export default {
  name: "ImageFusion",
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
      progressTimer: null     // 添加进度查询定时器
    }
  },
  computed: {
    canFusion() {
      return this.visibleImageUrl && this.infraredImageUrl && !this.loading;
    }
  },
  methods: {
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
          
          if (response.data.status === 'completed') {
            clearInterval(this.progressTimer);
            this.resultImageUrl = import.meta.env.VITE_APP_BASE_API + response.data.result;
            this.$modal.msgSuccess("图像融合成功");
            this.loading = false;
          } else if (response.data.status === 'failed') {
            clearInterval(this.progressTimer);
            this.$modal.msgError("图像融合失败");
            this.loading = false;
          }
        } catch (error) {
          console.log("error => ", error)
          clearInterval(this.progressTimer);
          this.$modal.msgError("获取进度失败");
          this.loading = false;
        }
      }, 1000);  // 每秒查询一次进度
    },
    
    // 在组件销毁时清除定时器
    beforeDestroy() {
      if (this.progressTimer) {
        clearInterval(this.progressTimer);
      }
    },
    
    // 重置时也要清除定时器
    handleReset() {
      if (this.progressTimer) {
        clearInterval(this.progressTimer);
      }
      this.visibleImageUrl = '';
      this.infraredImageUrl = '';
      this.resultImageUrl = '';
      this.loading = false;
      this.progress = 0;
      this.taskId = null;
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
  max-width: 100%;
  height: auto;
}
</style>