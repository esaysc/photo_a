<template>
  <div class="app-container">
    <!-- 搜索区保持不变 -->
    <el-form
      :model="queryParams"
      ref="queryForm"
      :inline="true"
      label-width="80px"
      class="search-form"
    >
      <el-form-item label="书名">
        <el-input
          v-model="queryParams.name"
          placeholder="请输入书籍名称"
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

    <!-- 修改卡片列表样式 -->
    <el-row :gutter="20" class="card-list">
      <el-col
        v-for="book in bookList"
        :key="book.id"
        :xs="24" :sm="12" :md="8" :lg="6"
      >
        <el-card
          class="book-card"
          shadow="hover"
          @click.native="onCardClick(book)"
        >
          <div class="cover-wrapper">
            <img
              class="book-cover"
              :src="book.coverPath || defaultCover"
              alt="封面"
            />
            <i class="el-icon-reading cover-icon"></i>
          </div>
          <div class="info">
            <h4 class="title">{{ book.name }}</h4>
            <p class="meta">
              <span>类型：{{ book.fileType }}</span>
              <span>人群：{{ book.audience }}</span>
            </p>
            <el-button
              class="download-btn"
              type="primary"
              size="mini"
              icon="el-icon-download"
              @click.stop="onDownload(book)"
            >下载</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 分页部分保持不变 -->
    <el-pagination
      v-if="total > 0"
      class="pagination"
      background
      :current-page="queryParams.pageNum"
      :page-size="queryParams.pageSize"
      :page-sizes="[10, 20, 50, 100]"
      :total="total"
      show-size-picker
      layout="total, sizes, prev, pager, next, jumper"
      @size-change="onSizeChange"
      @current-change="onPageChange"
    />
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { listBook } from '@/api/cms/book'
const baseUrl = 'http://localhost:8080';
const defaultCover = baseUrl+ '/profile/image/book-placeholder.jpg' // 你可以换成自己的默认封面

// 查询参数
const queryParams = reactive({
  pageNum: 1,
  pageSize: 12,
  name: '',
  audience: ''
})

const bookList = ref([])
const total = ref(0)

async function fetchBooks() {
  const { rows, total: t } = await listBook(queryParams)
  bookList.value = rows
  total.value = t
}

// 搜索
function onSearch() {
  queryParams.pageNum = 1
  fetchBooks()
}

// 重置
function onReset() {
  queryParams.name = ''
  queryParams.audience = ''
  onSearch()
}

// 点击卡片：可以跳转详情页或直接下载
function onCardClick(book) {
  // 例如跳转到详情页：
  // router.push({ name: 'BookDetail', params: { id: book.id } })
  // 这里简单做下载：
  window.open(book.storagePath, '_blank')
}

// 下载按钮
function onDownload(book) {
  const a = document.createElement('a')
  a.href = book.storagePath
  a.download = ''
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// 分页
function onPageChange(page) {
  queryParams.pageNum = page
  fetchBooks()
}
function onSizeChange(size) {
  queryParams.pageSize = size
  queryParams.pageNum = 1
  fetchBooks()
}

onMounted(fetchBooks)
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

.book-card {
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: transform .2s;
}

.book-card:hover {
  transform: translateY(-4px);
}

.cover-wrapper {
  position: relative;
}

.book-cover {
  width: 100%;
  height: 160px;
  object-fit: cover;
  border-radius: 4px 4px 0 0;
}

.cover-icon {
  position: absolute;
  font-size: 36px;
  color: rgba(255,255,255,0.8);
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  opacity: 0;
  transition: opacity .2s;
}

.book-card:hover .cover-icon {
  opacity: 1;
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
  width: 100%;
}

.pagination {
  text-align: center;
}
</style>
