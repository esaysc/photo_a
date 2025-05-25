<template>
  <div class="app-container">
    <div class="video-detail">
      <!-- 返回按钮 -->
      <el-button 
        icon="el-icon-back" 
        @click="goBack"
        class="back-btn"
      >返回列表</el-button>

      <!-- 视频信息 -->
      <div class="video-info">
        <h2 class="video-title">{{ video?.name }}</h2>
        <p class="video-meta">
          <span>时长：{{ video?.duration }}</span>
          <span>适用人群：{{ video?.audience }}</span>
        </p>
      </div>

      <!-- 视频播放器 -->
      <div class="player-wrapper">
        <video
          ref="videoPlayer"
          class="video-player"
          :src="video?.storagePath"
          controls
          autoplay
        ></video>
      </div>

      <!-- 下载按钮 -->
      <el-button
        type="primary"
        icon="el-icon-download"
        @click="download"
        class="download-btn"
      >下载视频</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getVideo } from '@/api/cms/video'

const route = useRoute()
const router = useRouter()
const video = ref(null)
const videoPlayer = ref(null)

// 获取视频详情
async function fetchVideoDetail() {
  const id = route.params.id
  try {
    const response = await getVideo(id)
    video.value = response.data
  } catch (error) {
    console.error('获取视频详情失败:', error)
  }
}

// 返回列表页
function goBack() {
  router.back()
}

// 下载视频
function download() {
  if (!video.value?.storagePath) return
  const a = document.createElement('a')
  a.href = video.value.storagePath
  a.download = video.value.name
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

onMounted(() => {
  fetchVideoDetail()
})
</script>

<style scoped>
.app-container {
  padding: 20px;
  background: #f5f5f5;
}

.video-detail {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

.back-btn {
  margin-bottom: 20px;
}

.video-info {
  margin-bottom: 20px;
}

.video-title {
  margin: 0 0 10px 0;
  font-size: 24px;
  color: #303133;
}

.video-meta {
  color: #666;
  font-size: 14px;
}

.video-meta span {
  margin-right: 20px;
}

.player-wrapper {
  margin: 20px 0;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
}

.video-player {
  width: 100%;
  max-height: 70vh;
}

.download-btn {
  margin-top: 20px;
}
</style>