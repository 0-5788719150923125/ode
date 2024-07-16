import * as tf from '@tensorflow/tfjs'
import LayerBase from './layers/base.js'
import { randomString } from './utils.js'
import AdaptiveMixtureOfExperts from './layers/AdaptiveMixtureOfExperts.js'
import Antirectifier from './layers/Antirectifier.js'
import AttentionFreeTransformer from './layers/AttentionFreeTransformer.js'
import Autoencoder from './layers/Autoencoder.js'
import CapsNet from './layers/CapsNet.js'
import ChunkedSelfAttention from './layers/ChunkedSelfAttention.js'
import DeterministicEmbedding from './layers/DeterministicEmbedding.js'
import EfficientAttention from './layers/EfficientAttention.js'
import EfficientChannelAttention from './layers/EfficientChannelAttention.js'
import FastAssociativeMemory from './layers/FastAssociativeMemory.js'
import FourierFeaturePositionalEncoding from './layers/FourierFeaturePositionalEncoding.js'
import GPT2Attention from './layers/GPT2Attention.js'
import GatedLinearMLP from './layers/GatedLinearMLP.js'
import GroupedQueryAttention from './layers/GroupedQueryAttention.js'
import IncrementalPowerIterationPCA from './layers/IncrementalPowerIterationPCA.js'
import IndependentComponentAnalysis from './layers/IndependentComponentAnalysis.js'
import LambdaLayer from './layers/LambdaLayer.js'
import ConstantSelfAttention from './layers/ConstantSelfAttention.js'
import MixtureOfDepths from './layers/MixtureOfDepths.js'
import MixtureOfExperts from './layers/MixtureOfExperts.js'
import MultiHeadAttention from './layers/MultiHeadAttention.js'
import MultiLayerPerceptron from './layers/MultiLayerPerceptron.js'
import MultiQueryAttention from './layers/MultiQueryAttention.js'
import ProjectedFeatureAttention from './layers/ProjectedFeatureAttention.js'
import ProjectedFeatureReduction from './layers/ProjectedFeatureReduction.js'
import QSparseMLP from './layers/QSparseMLP.js'
import Range from './layers/Range.js'
import RotaryPositionalEncoding from './layers/RotataryPositionalEncoding.js'
import SoftMergingOfExperts from './layers/SoftMergingOfExperts.js'
import SelfAttention from './layers/SelfAttention.js'
import SharedEmbedding from './layers/SharedEmbedding.js'
import SinusoidalPositionalEncoding from './layers/SinusoidalPositionalEncoding.js'
import SparseMixtureOfExperts from './layers/SparseMixtureOfExperts.js'
import SqueezeAndExcitation from './layers/SqueezeAndExcitation.js'
import StateSpace from './layers/StateSpace.js'
import SwarmOfExperts from './layers/SwarmOfExperts.js'
import SynthesizerAttention from './layers/SynthesizerAttention.js'
import VariableDimensionMLP from './layers/VariableDimensionMLP.js'
import VarianceThreshold from './layers/VarianceThreshold.js'
import LowRankFactorization from './layers/LowRankFactorization.js'

/**
 * @template {new (config: any) => import('@tensorflow/tfjs').layers.Layer} T
 * @param {T | ((config: any) => import('@tensorflow/tfjs').layers.Layer)} layerConstructor
 * @param {string} namePrefix
 * @returns {(config: any) => import('@tensorflow/tfjs').layers.Layer}
 */
const createLayerFactory = (layerConstructor, namePrefix) => (config) => {
    const name = `${namePrefix}-${randomString()}`
    if (typeof layerConstructor === 'function' && !layerConstructor.prototype) {
        // For TFJS layers which are created via functions
        return layerConstructor({ name, ...config })
    } else {
        // For custom layers which need 'new' keyword
        return new layerConstructor({ name, ...config })
    }
}

const customLayersConfig = {
    activation: { constructor: tf.layers.activation, prefix: 'act' },
    add: { constructor: tf.layers.add, prefix: 'add' },
    AdaptiveMixtureOfExperts: {
        constructor: AdaptiveMixtureOfExperts,
        prefix: 'moe'
    },
    Antirectifier: {
        constructor: Antirectifier,
        prefix: 'op'
    },
    AttentionFreeTransformer: {
        constructor: AttentionFreeTransformer,
        prefix: 'attn'
    },
    Autoencoder: {
        constructor: Autoencoder,
        prefix: 'dia'
    },
    bottleneck: { constructor: tf.layers.dense, prefix: 'bot' },
    CapsNet: {
        constructor: CapsNet,
        prefix: 'cap'
    },
    ChunkedSelfAttention: {
        constructor: ChunkedSelfAttention,
        prefix: 'attn'
    },
    concatenate: { constructor: tf.layers.concatenate, prefix: 'con' },
    ConstantSelfAttention: {
        constructor: ConstantSelfAttention,
        prefix: 'attn'
    },
    conv1d: { constructor: tf.layers.conv1d, prefix: 'c1d' },
    conv2d: { constructor: tf.layers.conv2d, prefix: 'c2d' },
    dense: { constructor: tf.layers.dense, prefix: 'ffd' },
    DeterministicEmbedding: {
        constructor: DeterministicEmbedding,
        prefix: 'emb'
    },
    EfficientAttention: { constructor: EfficientAttention, prefix: 'attn' },
    EfficientChannelAttention: {
        constructor: EfficientChannelAttention,
        prefix: 'attn'
    },
    embedding: { constructor: tf.layers.embedding, prefix: 'emb' },
    FastAssociativeMemory: {
        constructor: FastAssociativeMemory,
        prefix: 'mem'
    },
    FourierFeaturePositionalEncoding: {
        constructor: FourierFeaturePositionalEncoding,
        prefix: 'enc'
    },
    GatedLinearMLP: { constructor: GatedLinearMLP, prefix: 'mlp' },
    GPT2Attention: {
        constructor: GPT2Attention,
        prefix: 'attn'
    },
    GroupedQueryAttention: {
        constructor: GroupedQueryAttention,
        prefix: 'attn'
    },
    gru: { constructor: tf.layers.gru, prefix: 'gru' },
    IncrementalPowerIterationPCA: {
        constructor: IncrementalPowerIterationPCA,
        prefix: 'pca'
    },
    IndependentComponentAnalysis: {
        constructor: IndependentComponentAnalysis,
        prefix: 'ica'
    },
    input: { constructor: tf.layers.input, prefix: 'inp' },
    LambdaLayer: {
        constructor: LambdaLayer,
        prefix: 'op'
    },
    LowRankFactorization: {
        constructor: LowRankFactorization,
        prefix: 'op'
    },
    lstm: { constructor: tf.layers.lstm, prefix: 'lstm' },
    MixtureOfDepths: {
        constructor: MixtureOfDepths,
        prefix: 'moe'
    },
    MixtureOfExperts: {
        constructor: MixtureOfExperts,
        prefix: 'moe'
    },
    MultiHeadAttention: {
        constructor: MultiHeadAttention,
        prefix: 'attn'
    },
    MultiLayerPerceptron: { constructor: MultiLayerPerceptron, prefix: 'mlp' },
    multiply: { constructor: tf.layers.multiply, prefix: 'mul' },
    MultiQueryAttention: {
        constructor: MultiQueryAttention,
        prefix: 'attn'
    },
    ProjectedFeatureAttention: {
        constructor: ProjectedFeatureAttention,
        prefix: 'attn'
    },
    ProjectedFeatureReduction: {
        constructor: ProjectedFeatureReduction,
        prefix: 'op'
    },
    QSparseMLP: {
        constructor: QSparseMLP,
        prefix: 'mlp'
    },
    Range: {
        constructor: Range,
        prefix: 'op'
    },
    rnn: { constructor: tf.layers.gru, prefix: 'rnn' },
    RotaryPositionalEncoding: {
        constructor: RotaryPositionalEncoding,
        prefix: 'enc'
    },
    SelfAttention: { constructor: SelfAttention, prefix: 'attn' },
    SharedEmbedding: { constructor: SharedEmbedding, prefix: 'emb' },
    SinusoidalPositionalEncoding: {
        constructor: SinusoidalPositionalEncoding,
        prefix: 'enc'
    },
    SoftMergingOfExperts: { constructor: SoftMergingOfExperts, prefix: 'moe' },
    SparseMixtureOfExperts: {
        constructor: SparseMixtureOfExperts,
        prefix: 'moe'
    },
    SqueezeAndExcitation: {
        constructor: SqueezeAndExcitation,
        prefix: 'op'
    },
    StateSpace: {
        constructor: StateSpace,
        prefix: 'ssm'
    },
    SwarmOfExperts: {
        constructor: SwarmOfExperts,
        prefix: 'moe'
    },
    SynthesizerAttention: {
        constructor: SynthesizerAttention,
        prefix: 'attn'
    },
    timeDistributed: { constructor: tf.layers.timeDistributed, prefix: 'time' },
    VariableDimensionMLP: {
        constructor: VariableDimensionMLP,
        prefix: 'mlp'
    },
    VarianceThreshold: {
        constructor: VarianceThreshold,
        prefix: 'op'
    }
}

/**
 * @type {{[K in keyof typeof customLayersConfig]: (config: any) => LayerBase}}
 */
const customLayers = Object.fromEntries(
    Object.entries(customLayersConfig).map(([key, { constructor, prefix }]) => [
        key,
        createLayerFactory(constructor, prefix)
    ])
)

export default customLayers
