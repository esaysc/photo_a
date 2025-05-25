<template>
  <div class="app-container">
    <!-- 搜索表单 -->
    <el-form
      :model="queryParams"
      ref="queryForm"
      :inline="true"
      label-width="80px"
      class="search-form"
    >
      <el-form-item label="视频标题">
        <el-input
          v-model="queryParams.name"
          placeholder="请输入视频标题"
          clearable
          @keyup.enter.native="onSearch"
        />
      </el-form-item>
      <el-form-item label="适用人群">
        <el-input
          v-model="queryParams.audience"
          placeholder="请输入适用人群"
          clearable
          @keyup.enter.native="onSearch"
        />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" icon="el-icon-search" @click="onSearch">搜索</el-button>
        <el-button icon="el-icon-refresh" @click="onReset">重置</el-button>
      </el-form-item>
    </el-form>

    <!-- 卡片列表 -->
    <el-row :gutter="20" class="card-list">
      <el-col
        v-for="video in videoList"
        :key="video.id"
        :xs="24" :sm="12" :md="8" :lg="6"
      >
        <el-card
          class="video-card"
          shadow="hover"
          @click.native="openPlayer(video)"
        >
          <div class="cover-wrapper">
            <img
              class="video-cover"
              :src="video.coverPath || defaultCover"
              alt="封面"
            />
            <i class="el-icon-video-play play-icon"></i>
          </div>
          <div class="info">
            <h4 class="title">{{ video.name }}</h4>
            <p class="meta">
              <span>时长：{{ video.duration }}</span>
              <span>人群：{{ video.audience }}</span>
            </p>
            <el-button
              class="download-btn"
              type="primary"
              size="mini"
              icon="el-icon-download"
              @click.stop="download(video)"
            >下载</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 分页 -->
    <el-pagination
      v-if="total > 0"
      class="pagination"
      background
      :current-page="queryParams.pageNum"
      :page-size="queryParams.pageSize"
      :page-sizes="[12, 24, 48]"
      :total="total"
      layout="total, sizes, prev, pager, next, jumper"
      @size-change="onSizeChange"
      @current-change="onPageChange"
    />

    <!-- 播放器 Dialog -->
    <el-dialog
      :visible.sync="playerOpen"
      width="60%"
      :before-close="()=> playerOpen=false"
      center
    >
      <video
        ref="videoPlayer"
        class="video-player"
        :src="currentVideo?.storagePath"
        controls
        autoplay
      ></video>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { listVideo } from '@/api/cms/video'

const baseUrl = 'http://localhost:8080';
const defaultCover = baseUrl + '/profile/image/book-placeholder.jpg' 

const queryParams = reactive({
  pageNum: 1,
  pageSize: 12,
  name: '',
  audience: ''
})

const videoList = ref([])
const total = ref(0)
const playerOpen = ref(false)
const currentVideo = ref(null)

// 拉取数据
async function fetchVideos() {
  const { rows, total: t } = await listVideo(queryParams)
  videoList.value = rows
  total.value = t
}

// 搜索/重置
function onSearch() {
  queryParams.pageNum = 1
  fetchVideos()
}
function onReset() {
  queryParams.name = ''
  queryParams.audience = ''
  onSearch()
}

// 分页
function onPageChange(page) {
  queryParams.pageNum = page
  fetchVideos()
}
function onSizeChange(size) {
  queryParams.pageSize = size
  queryParams.pageNum = 1
  fetchVideos()
}

// 点击卡片，打开播放器
function openPlayer(video) {
  currentVideo.value = video
  playerOpen.value = true
}

// 下载
function download(video) {
  const a = document.createElement('a')
  a.href = video.storagePath
  a.download = ''
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

onMounted(fetchVideos)
</script>

<style scoped>
.app-container {
  padding: 20px;
  background: #f5f5f5;
}
.search-form {
  margin-bottom: 20px;
}
.card-list {
  margin-bottom: 20px;
}
.video-card {
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: transform .2s;
}
.video-card:hover {
  transform: translateY(-4px);
}
.cover-wrapper {
  position: relative;
}
.video-cover {
  width: 100%;
  height: 160px;
  object-fit: cover;
  border-radius: 4px 4px 0 0;
}
.play-icon {
  position: absolute;
  font-size: 36px;
  color: rgba(255,255,255,0.8);
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
}
.info {
  padding: 10px;
  background: #fff;
}
.title {
  margin: 0;
  font-size: 16px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.meta {
  margin: 6px 0;
  font-size: 12px;
  color: #666;
  display: flex;
  justify-content: space-between;
}
.download-btn {
  margin-top: 6px;
}
.pagination {
  text-align: center;
}
.video-player {
  width: 100%;
  height: 60vh;
  background: #000;
}
</style>