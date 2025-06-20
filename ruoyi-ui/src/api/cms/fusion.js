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

// 获取算法参数
export function getAlgorithmParams(fusionMethod) {
  return request({
    url: `/cms/fusion/params/${fusionMethod}`,
    method: 'get'
  })
}

// 获取算法执行计数
export function getAlgorithmCount(fusionMethod) {
  return request({
    url: `/cms/fusion/count/${fusionMethod}`,
    method: 'get'
  })
}

// 重置算法计数器
export function resetAlgorithmCounter(fusionMethod) {
  return request({
    url: `/cms/fusion/reset-counter/${fusionMethod}`,
    method: 'post'
  })
}

// 获取所有算法的计数统计
export function getAllCountStats() {
  return request({
    url: `/cms/fusion/count-stats`,
    method: 'get'
  })
}