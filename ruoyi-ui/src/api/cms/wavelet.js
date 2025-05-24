import request from '@/utils/request'

// 执行图像融合
export function fusionImages(data) {
  return request({
    url: '/cms/fusion/process',  // 更新为新的接口路径
    method: 'post',
    data: data
  })
}

// 获取融合历史记录列表
export function listFusionHistory(query) {
  return request({
    url: '/cms/wavelet/history/list',
    method: 'get',
    params: query
  })
}

// 获取融合历史记录详细信息
export function getFusionHistory(id) {
  return request({
    url: '/cms/wavelet/history/' + id,
    method: 'get'
  })
}

// 删除融合历史记录
export function delFusionHistory(id) {
  return request({
    url: '/cms/wavelet/history/' + id,
    method: 'delete'
  })
}