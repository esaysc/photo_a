<template>
  <div class="params-container">
    <div class="params-title">算法参数</div>
    <el-descriptions :column="column" border>
      <el-descriptions-item 
        v-for="param in algorithmParams" 
        :key="param.key" 
        :label="param.label"
      >
        {{ param.value }}
      </el-descriptions-item>
    </el-descriptions>
  </div>
</template>

<script>
import { getAlgorithmParams } from "@/api/cms/fusion";

export default {
  name: "AlgorithmParams",
  props: {
    // 融合方法
    fusionMethod: {
      type: String,
      required: true
    },
    // 展示列数
    column: {
      type: Number,
      default: 3
    }
  },
  data() {
    return {
      algorithmParams: []
    }
  },
  watch: {
    fusionMethod: {
      handler(newMethod) {
        if (newMethod) {
          this.loadAlgorithmParams();
        }
      },
      immediate: true
    }
  },
  methods: {
    // 加载算法参数
    async loadAlgorithmParams() {
      try {
        const response = await getAlgorithmParams(this.fusionMethod);
        if (response.code === 200) {
          this.algorithmParams = response.data || this.getDefaultParams();
        } else {
          // 如果接口失败，使用默认参数
          this.algorithmParams = this.getDefaultParams();
        }
      } catch (error) {
        console.error('获取算法参数失败:', error);
        // 使用默认参数
        this.algorithmParams = this.getDefaultParams();
      }
    },
    
    // 获取默认参数（根据算法类型）
    getDefaultParams() {
      const paramsMap = {
        'fast': [
          { key: 'method', label: '融合方法', value: '最优寻找方法' },
          { key: 'description', label: '算法描述', value: '快速融合算法，基于像素级的直接融合' },
          { key: 'complexity', label: '算法复杂度', value: 'O(n)' },
          { key: 'performance', label: '性能特点', value: '速度快，适用于实时应用' }
        ],
        'wavelet': [
          { key: 'method', label: '融合方法', value: '小波变换融合' },
          { key: 'waveletType', label: '小波基函数', value: 'db3' },
          { key: 'decompositionLevel', label: '分解层数', value: '2' },
          { key: 'lowFreqRule', label: '低频融合规则', value: '取平均值' },
          { key: 'highFreqRule', label: '高频融合规则', value: '梯度加权融合' },
          { key: 'enhancement', label: '增强处理', value: '非锐化掩膜增强' },
          { key: 'blurKernel', label: '模糊核大小', value: '5x5' }
        ],
        'pyramid': [
          { key: 'method', label: '融合方法', value: '拉普拉斯金字塔融合' },
          { key: 'levels', label: '金字塔层数', value: '4' },
          { key: 'pyramidType', label: '金字塔类型', value: '高斯+拉普拉斯' },
          { key: 'fusionRule', label: '融合规则', value: '基于方差的高频加权' },
          { key: 'varianceWindow', label: '方差计算窗口', value: '3x3' },
          { key: 'edgeEnhancement', label: '边缘增强', value: 'Canny边缘检测' },
          { key: 'cannyThreshold', label: 'Canny阈值', value: '50-150' },
          { key: 'edgeWeight', label: '边缘权重', value: '0.3' }
        ],
        'sparse': [
          { key: 'method', label: '融合方法', value: '稀疏表示融合' },
          { key: 'dependency', label: '依赖库', value: 'scikit-learn' },
          { key: 'dictionarySize', label: '字典大小', value: '256' },
          { key: 'patchSize', label: '分块大小', value: '8x8' },
          { key: 'sparsityLevel', label: '稀疏度', value: '0.1' }
        ],
        'cnn': [
          { key: 'method', label: '融合方法', value: '卷积神经网络融合' },
          { key: 'dependency', label: '依赖库', value: 'PyTorch' },
          { key: 'modelFile', label: '模型文件', value: 'cnn_fusion_model.pth' },
          { key: 'architecture', label: '网络架构', value: '深度卷积网络' },
          { key: 'inputChannels', label: '输入通道', value: '6 (可见光3+红外3)' },
          { key: 'outputChannels', label: '输出通道', value: '3 (RGB)' }
        ],
        'self-encoder': [
          { key: 'method', label: '融合方法', value: '自编码器融合' },
          { key: 'dependency', label: '依赖库', value: 'PyTorch' },
          { key: 'architecture', label: '网络架构', value: '编码器-解码器' },
          { key: 'encoderLayers', label: '编码器层数', value: '多层卷积编码' },
          { key: 'decoderLayers', label: '解码器层数', value: '多层卷积解码' },
          { key: 'latentSpace', label: '潜在空间', value: '压缩特征表示' }
        ],
        'gan': [
          { key: 'method', label: '融合方法', value: '生成对抗网络融合' },
          { key: 'dependency', label: '依赖库', value: 'PyTorch' },
          { key: 'generator', label: '生成器', value: '卷积生成网络' },
          { key: 'discriminator', label: '判别器', value: '卷积判别网络' },
          { key: 'lossFunction', label: '损失函数', value: '对抗损失+内容损失' },
          { key: 'trainingMode', label: '训练模式', value: '对抗训练' }
        ],
        'ganresnet': [
          { key: 'method', label: '融合方法', value: 'GAN-ResNet融合' },
          { key: 'dependency', label: '依赖库', value: 'PyTorch' },
          { key: 'generator', label: '生成器', value: 'ResNet + GAN' },
          { key: 'resnetBlocks', label: 'ResNet块', value: '残差连接块' },
          { key: 'skipConnections', label: '跳跃连接', value: 'ResNet残差连接' },
          { key: 'discriminator', label: '判别器', value: '卷积判别网络' },
          { key: 'architecture', label: '网络架构', value: 'ResNet + 对抗网络' }
        ]
      };
      
      return paramsMap[this.fusionMethod] || [
        { key: 'method', label: '融合方法', value: this.fusionMethod },
        { key: 'status', label: '状态', value: '未实现的算法' }
      ];
    }
  }
}
</script>

<style scoped>
.params-container {
  background-color: #f5f7fa;
  padding: 20px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.params-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 15px;
  color: #303133;
}
</style> 