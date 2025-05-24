<template>
  <div class="app-container">
    <!-- 搜索区 -->
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

    <!-- 卡片列表 -->
    <el-row :gutter="20" class="card-list">
      <el-col
        v-for="book in bookList"
        :key="book.id"
        :xs="24"
        :sm="12"
        :md="8"
        :lg="6"
      >
        <el-card
          class="book-card"
          :body-style="{ padding: '10px' }"
          shadow="hover"
          @click.native="onCardClick(book)"
        >
          <img
            class="book-cover"
            :src="book.coverPath || defaultCover"
            alt="封面"
          />
          <h3 class="book-title">{{ book.name }}</h3>
          <p class="book-info">类型：{{ book.fileType }}</p>
          <p class="book-info">人群：{{ book.audience }}</p>
          <el-button
            class="download-btn"
            type="primary"
            size="small"
            icon="el-icon-download"
            @click.stop="onDownload(book)"
          >
            下载
          </el-button>
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

const defaultCover = '/static/book-placeholder.png' // 你可以换成自己的默认封面

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
  background: #f9f9f9;
}

.search-form {
  margin-bottom: 20px;
}

.card-list {
  margin-bottom: 20px;
}

.book-card {
  cursor: pointer;
  transition: transform .2s;
}
.book-card:hover {
  transform: translateY(-5px);
}

.book-cover {
  width: 100%;
  height: 180px;
  object-fit: cover;
  border-radius: 4px;
}

.book-title {
  margin: 10px 0 5px;
  font-size: 16px;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.book-info {
  margin: 2px 0;
  color: #666;
  font-size: 13px;
}

.download-btn {
  margin-top: 8px;
}

.pagination {
  text-align: center;
}
</style>