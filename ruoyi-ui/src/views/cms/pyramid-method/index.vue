<template>
  <image-fusion
    title="基于金字塔的图像融合方法"
    fusion-method="pyramid"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion/index.vue'

export default {
  name: "PyramidMethod",
  components: {
    ImageFusion
  }
}
</script>