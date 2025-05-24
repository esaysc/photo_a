<template>
  <image-fusion
    title="基于稀疏表示的图像融合方法"
    fusion-method="sparse"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion/index.vue'

export default {
  name: 'SparseMethod',
  components: {
    ImageFusion
  }
}
</script>