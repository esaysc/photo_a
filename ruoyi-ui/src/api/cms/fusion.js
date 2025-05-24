import request from '@/utils/request'

// 图像融合处理
export function fusionImages(data) {
  return request({
    url: '/cms/fusion/process',
    method: 'post',
    data: data
  })
}

export function getFusionProgress(taskId) {
  return request({
    url: `/cms/fusion/progress/${taskId}`,
    method: 'get'
  })
}