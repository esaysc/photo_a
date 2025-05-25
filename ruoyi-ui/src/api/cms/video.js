import request from '@/utils/request'

// 查询视频列表
export function listVideo(query) {
  return request({
    url: '/cms/video/list',
    method: 'get',
    params: query
  })
}

// 查询视频详情
export function getVideo(id) {
  return request({
    url: '/cms/video/' + id,
    method: 'get'
  })
}

// 新增视频
export function addVideo(data) {
  return request({
    url: '/cms/video',
    method: 'post',
    data: data
  })
}

// 修改视频
export function updateVideo(data) {
  return request({
    url: '/cms/video',
    method: 'put',
    data: data
  })
}

// 删除视频
export function delVideo(id) {
  return request({
    url: '/cms/video/' + id,
    method: 'delete'
  })
}
