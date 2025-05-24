<template>
  <ImageFusion
    title="GAN-ResNet图像融合"
    fusion-method="ganresnet"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion'

export default {
  name: 'GanResnet',
  components: {
    ImageFusion
  }
}
</script>