<template>
  <div class="app-container">
    <el-form :model="queryParams" ref="queryRef" :inline="true" v-show="showSearch" label-width="68px">
      <el-form-item label="书籍名称" prop="name">
        <el-input
          v-model="queryParams.name"
          placeholder="请输入书籍名称"
          clearable
          @keyup.enter="handleQuery"
        />
      </el-form-item>
      <el-form-item label="适用人群" prop="audience">
        <el-input
          v-model="queryParams.audience"
          placeholder="请输入适用人群"
          clearable
          @keyup.enter="handleQuery"
        />
      </el-form-item>
      <el-form-item label="关联 book_category" prop="categoryId">
        <el-input
          v-model="queryParams.categoryId"
          placeholder="请输入关联 book_category"
          clearable
          @keyup.enter="handleQuery"
        />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" icon="Search" @click="handleQuery">搜索</el-button>
        <el-button icon="Refresh" @click="resetQuery">重置</el-button>
      </el-form-item>
    </el-form>

    <el-row :gutter="10" class="mb8">
      <el-col :span="1.5">
        <el-button
          type="primary"
          plain
          icon="Plus"
          @click="handleAdd"
          v-hasPermi="['cms:book_resource:add']"
        >新增</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="success"
          plain
          icon="Edit"
          :disabled="single"
          @click="handleUpdate"
          v-hasPermi="['cms:book_resource:edit']"
        >修改</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="danger"
          plain
          icon="Delete"
          :disabled="multiple"
          @click="handleDelete"
          v-hasPermi="['cms:book_resource:remove']"
        >删除</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="warning"
          plain
          icon="Download"
          @click="handleExport"
          v-hasPermi="['cms:book_resource:export']"
        >导出</el-button>
      </el-col>
      <right-toolbar v-model:showSearch="showSearch" @queryTable="getList"></right-toolbar>
    </el-row>

    <el-table v-loading="loading" :data="book_resourceList" @selection-change="handleSelectionChange">
      <el-table-column type="selection" width="55" align="center" />
      <el-table-column label="书编号" align="center" prop="id" />
      <el-table-column label="书籍名称" align="center" prop="name" />
      <el-table-column label="文件存储路径" align="center" prop="storagePath" />
      <el-table-column label="文件类型，如 PDF、EPUB" align="center" prop="fileType" />
      <el-table-column label="书籍简介" align="center" prop="description" />
      <el-table-column label="适用人群" align="center" prop="audience" />
      <el-table-column label="关联 book_category" align="center" prop="categoryId" />
      <el-table-column label="操作" align="center" class-name="small-padding fixed-width">
        <template #default="scope">
          <el-button link type="primary" icon="Edit" @click="handleUpdate(scope.row)" v-hasPermi="['cms:book_resource:edit']">修改</el-button>
          <el-button link type="primary" icon="Delete" @click="handleDelete(scope.row)" v-hasPermi="['cms:book_resource:remove']">删除</el-button>
        </template>
      </el-table-column>
    </el-table>
    
    <pagination
      v-show="total>0"
      :total="total"
      v-model:page="queryParams.pageNum"
      v-model:limit="queryParams.pageSize"
      @pagination="getList"
    />

    <!-- 添加或修改图书资源对话框 -->
    <el-dialog :title="title" v-model="open" width="500px" append-to-body>
      <el-form ref="book_resourceRef" :model="form" :rules="rules" label-width="80px">
        <el-form-item label="书籍名称" prop="name">
          <el-input v-model="form.name" placeholder="请输入书籍名称" />
        </el-form-item>
        <el-form-item label="文件存储路径" prop="storagePath">
          <el-input v-model="form.storagePath" type="textarea" placeholder="请输入内容" />
        </el-form-item>
        <el-form-item label="书籍简介" prop="description">
          <el-input v-model="form.description" type="textarea" placeholder="请输入内容" />
        </el-form-item>
        <el-form-item label="适用人群" prop="audience">
          <el-input v-model="form.audience" placeholder="请输入适用人群" />
        </el-form-item>
        <el-form-item label="关联 book_category" prop="categoryId">
          <el-input v-model="form.categoryId" placeholder="请输入关联 book_category" />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="submitForm">确 定</el-button>
          <el-button @click="cancel">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup name="Book_resource">
import { listBook_resource, getBook_resource, delBook_resource, addBook_resource, updateBook_resource } from "@/api/cms/book_resource"

const { proxy } = getCurrentInstance()

const book_resourceList = ref([])
const open = ref(false)
const loading = ref(true)
const showSearch = ref(true)
const ids = ref([])
const single = ref(true)
const multiple = ref(true)
const total = ref(0)
const title = ref("")

const data = reactive({
  form: {},
  queryParams: {
    pageNum: 1,
    pageSize: 10,
    name: null,
    storagePath: null,
    fileType: null,
    description: null,
    audience: null,
    categoryId: null,
  },
  rules: {
    name: [
      { required: true, message: "书籍名称不能为空", trigger: "blur" }
    ],
    storagePath: [
      { required: true, message: "文件存储路径不能为空", trigger: "blur" }
    ],
    fileType: [
      { required: true, message: "文件类型，如 PDF、EPUB不能为空", trigger: "change" }
    ],
    categoryId: [
      { required: true, message: "关联 book_category不能为空", trigger: "blur" }
    ],
  }
})

const { queryParams, form, rules } = toRefs(data)

/** 查询图书资源列表 */
function getList() {
  loading.value = true
  listBook_resource(queryParams.value).then(response => {
    book_resourceList.value = response.rows
    total.value = response.total
    loading.value = false
  })
}

// 取消按钮
function cancel() {
  open.value = false
  reset()
}

// 表单重置
function reset() {
  form.value = {
    id: null,
    name: null,
    storagePath: null,
    fileType: null,
    description: null,
    audience: null,
    categoryId: null,
    createdAt: null
  }
  proxy.resetForm("book_resourceRef")
}

/** 搜索按钮操作 */
function handleQuery() {
  queryParams.value.pageNum = 1
  getList()
}

/** 重置按钮操作 */
function resetQuery() {
  proxy.resetForm("queryRef")
  handleQuery()
}

// 多选框选中数据
function handleSelectionChange(selection) {
  ids.value = selection.map(item => item.id)
  single.value = selection.length != 1
  multiple.value = !selection.length
}

/** 新增按钮操作 */
function handleAdd() {
  reset()
  open.value = true
  title.value = "添加图书资源"
}

/** 修改按钮操作 */
function handleUpdate(row) {
  reset()
  const _id = row.id || ids.value
  getBook_resource(_id).then(response => {
    form.value = response.data
    open.value = true
    title.value = "修改图书资源"
  })
}

/** 提交按钮 */
function submitForm() {
  proxy.$refs["book_resourceRef"].validate(valid => {
    if (valid) {
      if (form.value.id != null) {
        updateBook_resource(form.value).then(response => {
          proxy.$modal.msgSuccess("修改成功")
          open.value = false
          getList()
        })
      } else {
        addBook_resource(form.value).then(response => {
          proxy.$modal.msgSuccess("新增成功")
          open.value = false
          getList()
        })
      }
    }
  })
}

/** 删除按钮操作 */
function handleDelete(row) {
  const _ids = row.id || ids.value
  proxy.$modal.confirm('是否确认删除图书资源编号为"' + _ids + '"的数据项？').then(function() {
    return delBook_resource(_ids)
  }).then(() => {
    getList()
    proxy.$modal.msgSuccess("删除成功")
  }).catch(() => {})
}

/** 导出按钮操作 */
function handleExport() {
  proxy.download('cms/book_resource/export', {
    ...queryParams.value
  }, `book_resource_${new Date().getTime()}.xlsx`)
}

getList()
</script>
