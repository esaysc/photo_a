<template>
  <ImageFusion
    title="卷积神经网络图像融合"
    fusion-method="'cnn'"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion/index.vue'

export default {
  name: 'ConvolutionalNeuralNetwork',
  components: {
    ImageFusion
  }
}
</script>