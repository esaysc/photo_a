import request from '@/utils/request'

// 上传可见光图像
export function uploadVisibleImage(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request({
    url: '/common/upload',
    method: 'post',
    data: formData
  })
}

// 上传红外线光图像
export function uploadInfraredImage(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request({
    url: '/common/upload',
    method: 'post',
    data: formData
  })
}

// 执行图像融合
export function fusionImages(data) {
  return request({
    url: '/wavelet/fusion',
    method: 'post',
    data: data
  })
}