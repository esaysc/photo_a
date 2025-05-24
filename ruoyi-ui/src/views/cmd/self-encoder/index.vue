<template>
  <ImageFusion
    title="自编码器图像融合"
    fusion-method="self-encoder"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion/index.vue'

export default {
  name: 'SelfEncoder',
  components: {
    ImageFusion
  }
}
</script>