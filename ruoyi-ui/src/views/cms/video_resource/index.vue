<template>
  <div class="app-container">
    <el-form :model="queryParams" ref="queryRef" :inline="true" v-show="showSearch" label-width="68px">
      <el-form-item label="视频标题" prop="name">
        <el-input
          v-model="queryParams.name"
          placeholder="请输入视频标题"
          clearable
          @keyup.enter="handleQuery"
        />
      </el-form-item>
      <el-form-item label="视频时长" prop="duration">
        <el-date-picker clearable
          v-model="queryParams.duration"
          type="date"
          value-format="YYYY-MM-DD"
          placeholder="请选择视频时长">
        </el-date-picker>
      </el-form-item>
      <el-form-item label="适用人群" prop="audience">
        <el-input
          v-model="queryParams.audience"
          placeholder="请输入适用人群"
          clearable
          @keyup.enter="handleQuery"
        />
      </el-form-item>
      <el-form-item label="关联 video_category" prop="categoryId">
        <el-input
          v-model="queryParams.categoryId"
          placeholder="请输入关联 video_category"
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
          v-hasPermi="['cms:video_resource:add']"
        >新增</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="success"
          plain
          icon="Edit"
          :disabled="single"
          @click="handleUpdate"
          v-hasPermi="['cms:video_resource:edit']"
        >修改</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="danger"
          plain
          icon="Delete"
          :disabled="multiple"
          @click="handleDelete"
          v-hasPermi="['cms:video_resource:remove']"
        >删除</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="warning"
          plain
          icon="Download"
          @click="handleExport"
          v-hasPermi="['cms:video_resource:export']"
        >导出</el-button>
      </el-col>
      <right-toolbar v-model:showSearch="showSearch" @queryTable="getList"></right-toolbar>
    </el-row>

    <el-table v-loading="loading" :data="video_resourceList" @selection-change="handleSelectionChange">
      <el-table-column type="selection" width="55" align="center" />
      <el-table-column label="${comment}" align="center" prop="id" />
      <el-table-column label="视频标题" align="center" prop="name" />
      <el-table-column label="文件存储路径" align="center" prop="storagePath" />
      <el-table-column label="视频时长" align="center" prop="duration" width="180">
        <template #default="scope">
          <span>{{ parseTime(scope.row.duration, '{y}-{m}-{d}') }}</span>
        </template>
      </el-table-column>
      <el-table-column label="视频简介" align="center" prop="description" />
      <el-table-column label="适用人群" align="center" prop="audience" />
      <el-table-column label="封面图像路径" align="center" prop="coverPath" />
      <el-table-column label="关联 video_category" align="center" prop="categoryId" />
      <el-table-column label="操作" align="center" class-name="small-padding fixed-width">
        <template #default="scope">
          <el-button link type="primary" icon="Edit" @click="handleUpdate(scope.row)" v-hasPermi="['cms:video_resource:edit']">修改</el-button>
          <el-button link type="primary" icon="Delete" @click="handleDelete(scope.row)" v-hasPermi="['cms:video_resource:remove']">删除</el-button>
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

    <!-- 添加或修改视频资源对话框 -->
    <el-dialog :title="title" v-model="open" width="500px" append-to-body>
      <el-form ref="video_resourceRef" :model="form" :rules="rules" label-width="80px">
        <el-form-item label="视频标题" prop="name">
          <el-input v-model="form.name" placeholder="请输入视频标题" />
        </el-form-item>
        <el-form-item label="文件存储路径" prop="storagePath">
          <el-input v-model="form.storagePath" type="textarea" placeholder="请输入内容" />
        </el-form-item>
        <el-form-item label="视频时长" prop="duration">
          <el-date-picker clearable
            v-model="form.duration"
            type="date"
            value-format="YYYY-MM-DD"
            placeholder="请选择视频时长">
          </el-date-picker>
        </el-form-item>
        <el-form-item label="视频简介" prop="description">
          <el-input v-model="form.description" type="textarea" placeholder="请输入内容" />
        </el-form-item>
        <el-form-item label="适用人群" prop="audience">
          <el-input v-model="form.audience" placeholder="请输入适用人群" />
        </el-form-item>
        <el-form-item label="封面图像路径" prop="coverPath">
          <el-input v-model="form.coverPath" type="textarea" placeholder="请输入内容" />
        </el-form-item>
        <el-form-item label="关联 video_category" prop="categoryId">
          <el-input v-model="form.categoryId" placeholder="请输入关联 video_category" />
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

<script setup name="Video_resource">
import { listVideo_resource, getVideo_resource, delVideo_resource, addVideo_resource, updateVideo_resource } from "@/api/cms/video_resource"

const { proxy } = getCurrentInstance()

const video_resourceList = ref([])
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
    duration: null,
    description: null,
    audience: null,
    coverPath: null,
    categoryId: null,
  },
  rules: {
    name: [
      { required: true, message: "视频标题不能为空", trigger: "blur" }
    ],
    storagePath: [
      { required: true, message: "文件存储路径不能为空", trigger: "blur" }
    ],
    duration: [
      { required: true, message: "视频时长不能为空", trigger: "blur" }
    ],
    categoryId: [
      { required: true, message: "关联 video_category不能为空", trigger: "blur" }
    ],
  }
})

const { queryParams, form, rules } = toRefs(data)

/** 查询视频资源列表 */
function getList() {
  loading.value = true
  listVideo_resource(queryParams.value).then(response => {
    video_resourceList.value = response.rows
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
    duration: null,
    description: null,
    audience: null,
    coverPath: null,
    categoryId: null,
    createdAt: null
  }
  proxy.resetForm("video_resourceRef")
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
  title.value = "添加视频资源"
}

/** 修改按钮操作 */
function handleUpdate(row) {
  reset()
  const _id = row.id || ids.value
  getVideo_resource(_id).then(response => {
    form.value = response.data
    open.value = true
    title.value = "修改视频资源"
  })
}

/** 提交按钮 */
function submitForm() {
  proxy.$refs["video_resourceRef"].validate(valid => {
    if (valid) {
      if (form.value.id != null) {
        updateVideo_resource(form.value).then(response => {
          proxy.$modal.msgSuccess("修改成功")
          open.value = false
          getList()
        })
      } else {
        addVideo_resource(form.value).then(response => {
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
  proxy.$modal.confirm('是否确认删除视频资源编号为"' + _ids + '"的数据项？').then(function() {
    return delVideo_resource(_ids)
  }).then(() => {
    getList()
    proxy.$modal.msgSuccess("删除成功")
  }).catch(() => {})
}

/** 导出按钮操作 */
function handleExport() {
  proxy.download('cms/video_resource/export', {
    ...queryParams.value
  }, `video_resource_${new Date().getTime()}.xlsx`)
}

getList()
</script>
