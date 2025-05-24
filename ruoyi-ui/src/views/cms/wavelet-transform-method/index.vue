<template>
  <image-fusion
    title="基于小波变换的图像融合方法"
    fusion-method="wavelet"
  />
</template>

<script>
import ImageFusion from '@/components/ImageFusion'

export default {
  name: "WaveletTransform",
  components: {
    ImageFusion
  }
}
</script>