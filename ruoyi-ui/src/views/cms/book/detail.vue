<template>
  <div class="app-container">
    <el-card class="book-detail">
      <template #header>
        <div class="card-header">
          <el-page-header @back="goBack" :content="bookInfo.name || '图书详情'" />
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="cover-wrapper">
            <img :src="bookInfo.coverPath || defaultCover" :alt="bookInfo.name" class="book-cover" />
            <div class="book-meta">
              <p><strong>类型：</strong>{{ bookInfo.fileType }}</p>
              <p><strong>适用人群：</strong>{{ bookInfo.audience }}</p>
              <el-button 
                type="primary" 
                icon="el-icon-download"
                @click="onDownload"
                class="download-btn"
              >下载文件</el-button>
            </div>
          </div>
        </el-col>
        
        <el-col :span="18">
          <div class="content-wrapper">
            <!-- PDF预览 -->
            <template v-if="isPDF">
              <vue-pdf-embed
                :source="bookInfo.storagePath"
                :page="currentPage"
                class="pdf-viewer"
                @loaded="onPdfLoaded"
              />
              <div class="pdf-controls" v-if="totalPages > 0">
                <el-pagination
                  layout="prev, pager, next"
                  :total="totalPages"
                  :current-page.sync="currentPage"
                  :page-size="1"
                />
              </div>
            </template>
            
            <!-- Markdown预览 -->
            <div v-else-if="isMarkdown" class="markdown-content" v-html="renderedContent"></div>
            
            <!-- 图片预览 -->
            <div v-else-if="isImage" class="image-preview">
              <el-image 
                :src="bookInfo.storagePath"
                :preview-src-list="[bookInfo.storagePath]"
                fit="contain"
              />
            </div>
            
            <!-- 其他文件类型 -->
            <div v-else class="file-info">
              <el-empty description="该文件类型暂不支持在线预览，请下载后查看" />
            </div>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getBook } from '@/api/cms/book'
import VuePdfEmbed from 'vue-pdf-embed'
import { marked } from 'marked'

const route = useRoute()
const router = useRouter()
const baseUrl = 'http://localhost:8080'
const defaultCover = baseUrl + '/profile/image/book-placeholder.jpg'

// 图书信息
const bookInfo = ref({})

// PDF相关
const currentPage = ref(1)
const totalPages = ref(0)

// Markdown内容
const markdownContent = ref('')
const renderedContent = computed(() => {
  return marked(markdownContent.value)
})

// 文件类型判断
const isPDF = computed(() => bookInfo.value.fileType?.toLowerCase() === 'pdf')
const isMarkdown = computed(() => bookInfo.value.fileType?.toLowerCase() === 'md')
const isImage = computed(() => {
  const type = bookInfo.value.fileType?.toLowerCase()
  return ['jpg', 'jpeg', 'png', 'gif'].includes(type)
})

// 获取图书信息
async function fetchBookInfo() {
  try {
    console.log("route => ", route);
    
    const { data } = await getBook(route.params.id)
    bookInfo.value = data
    
    // 如果是Markdown文件，获取内容
    if (isMarkdown.value) {
      const response = await fetch(bookInfo.value.storagePath)
      markdownContent.value = await response.text()
    }
  } catch (error) {
    console.error('获取图书信息失败：', error)
    ElMessage.error('获取图书信息失败')
  }
}

// PDF加载完成
function onPdfLoaded(e) {
  totalPages.value = e.numberOfPages
}

// 返回上一页
function goBack() {
  router.back()
}

// 下载文件
function onDownload() {
  const a = document.createElement('a')
  a.href = bookInfo.value.storagePath
  a.download = bookInfo.value.name
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

onMounted(() => {
  fetchBookInfo()
})
</script>

<style scoped>
.app-container {
  padding: 20px;
}

.book-detail {
  min-height: calc(100vh - 120px);
}

.card-header {
  padding: 0;
}

.cover-wrapper {
  text-align: center;
}

.book-cover {
  width: 100%;
  max-width: 300px;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

.book-meta {
  margin-top: 20px;
  text-align: left;
}

.book-meta p {
  margin: 10px 0;
  color: #666;
}

.download-btn {
  width: 100%;
  margin-top: 20px;
}

.content-wrapper {
  min-height: 500px;
  padding: 20px;
  background: #fff;
  border-radius: 4px;
}

.pdf-viewer {
  width: 100%;
  min-height: 500px;
}

.pdf-controls {
  margin-top: 20px;
  text-align: center;
}

.markdown-content {
  padding: 20px;
  line-height: 1.6;
}

.image-preview {
  text-align: center;
}

.image-preview :deep(.el-image) {
  max-width: 100%;
  max-height: 70vh;
}

.file-info {
  padding: 40px;
  text-align: center;
}
</style>