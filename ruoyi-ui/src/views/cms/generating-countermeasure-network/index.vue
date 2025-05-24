<template>
  <ImageFusion
    title="生成对抗网络图像融合"
    fusion-method="gan"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion/index.vue'

export default {
  name: 'GeneratingCountermeasureNetwork',
  components: {
    ImageFusion
  }
}
</script>