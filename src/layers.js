import * as tf from '@tensorflow/tfjs'
import { randomString } from './utils.js'
import ConstantSelfAttention from './layers/ConstantSelfAttention.js'
import EfficientAttention from './layers/EfficientAttention.js'
import GatedLinearMLP from './layers/GatedLinearMLP.js'
import GroupedQueryAttention from './layers/GroupedQueryAttention.js'
import IndependentComponentAnalysis from './layers/IndependentComponentAnalysis.js'
import MultiHeadAttention from './layers/MultiHeadAttention.js'
import MultiLayerPerceptron from './layers/MultiLayerPerceptron.js'
import MultiQueryAttention from './layers/MultiQueryAttention.js'
import ProjectedFeatureAttention from './layers/ProjectedFeatureAttention.js'
import SMEAR from './layers/SMEAR.js'
import SelfAttention from './layers/SelfAttention.js'
import SharedEmbedding from './layers/SharedEmbedding.js'
import VariableDimensionMLP from './layers/VariableDimensionMLP.js'
import MixtureOfExperts from './layers/MixtureOfExperts.js'
import MixtureOfDepths from './layers/MixtureOfDepths.js'
import AdaptiveMixtureOfExperts from './layers/AdaptiveMixtureOfExperts copy.js'
import SparseMixtureOfExperts from './layers/SparseMixtureOfExperts.js'
import DeterministicEmbedding from './layers/DeterministicEmbedding.js'
import SwarmOfExperts from './layers/SwarmOfExperts.js'
import Autoencoder from './layers/Autoencoder.js'
import FastAssociativeMemory from './layers/FastAssociativeMemory.js'
import CapsNet from './layers/CapsNet.js'
import Range from './layers/Range.js'
import SinusoidalPositionalEncoding from './layers/SinusoidalPositionalEncoding.js'
import GPT2Attention from './layers/GPT2Attention.js'
import SynthesizerAttention from './layers/SynthesizerAttention.js'
import LambdaLayer from './layers/LambdaLayer.js'
import ProjectedFeatureReduction from './layers/ProjectedFeatureReduction.js'
import RotaryPositionalEncoding from './layers/RotataryPositionalEncoding.js'
import LinearAttention from './layers/LinearAttention.js'
import Antirectifier from './layers/Antirectifier.js'
import EfficientChannelAttention from './layers/EfficientChannelAttention.js'
import SqueezeAndExcitation from './layers/SqueezeAndExcitation.js'
import FourierFeaturePositionalEncoding from './layers/FourierFeaturePositionalEncoding.js'
import IncrementalPowerIterationPCA from './layers/IncrementalPowerIterationPCA.js'
import AttentionFreeTransformer from './layers/AttentionFreeTransformer.js'

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
    bottleneck: { constructor: tf.layers.dense, prefix: 'bot' },
    concatenate: { constructor: tf.layers.concatenate, prefix: 'con' },
    conv1d: { constructor: tf.layers.conv1d, prefix: 'c1d' },
    conv2d: { constructor: tf.layers.conv2d, prefix: 'c2d' },
    dense: { constructor: tf.layers.dense, prefix: 'ffd' },
    embedding: { constructor: tf.layers.embedding, prefix: 'emb' },
    input: { constructor: tf.layers.input, prefix: 'inp' },
    multiply: { constructor: tf.layers.multiply, prefix: 'mul' },
    timeDistributed: { constructor: tf.layers.timeDistributed, prefix: 'time' },
    gru: { constructor: tf.layers.gru, prefix: 'gru' },
    lstm: { constructor: tf.layers.lstm, prefix: 'lstm' },
    rnn: { constructor: tf.layers.gru, prefix: 'rnn' },
    SharedEmbedding: { constructor: SharedEmbedding, prefix: 'emb' },
    ProjectedFeatureAttention: {
        constructor: ProjectedFeatureAttention,
        prefix: 'attn'
    },
    SMEAR: { constructor: SMEAR, prefix: 'moe' },
    MultiLayerPerceptron: { constructor: MultiLayerPerceptron, prefix: 'mlp' },
    GatedLinearMLP: { constructor: GatedLinearMLP, prefix: 'mlp' },
    SelfAttention: { constructor: SelfAttention, prefix: 'attn' },
    EfficientAttention: { constructor: EfficientAttention, prefix: 'attn' },
    ConstantSelfAttention: {
        constructor: ConstantSelfAttention,
        prefix: 'attn'
    },
    MultiHeadAttention: {
        constructor: MultiHeadAttention,
        prefix: 'attn'
    },
    MultiQueryAttention: {
        constructor: MultiQueryAttention,
        prefix: 'attn'
    },
    GroupedQueryAttention: {
        constructor: GroupedQueryAttention,
        prefix: 'attn'
    },
    IndependentComponentAnalysis: {
        constructor: IndependentComponentAnalysis,
        prefix: 'ica'
    },
    VariableDimensionMLP: {
        constructor: VariableDimensionMLP,
        prefix: 'mlp'
    },
    MixtureOfExperts: {
        constructor: MixtureOfExperts,
        prefix: 'moe'
    },
    MixtureOfDepths: {
        constructor: MixtureOfDepths,
        prefix: 'moe'
    },
    SparseMixtureOfExperts: {
        constructor: SparseMixtureOfExperts,
        prefix: 'moe'
    },
    AdaptiveMixtureOfExperts: {
        constructor: AdaptiveMixtureOfExperts,
        prefix: 'moe'
    },
    DeterministicEmbedding: {
        constructor: DeterministicEmbedding,
        prefix: 'emb'
    },
    SwarmOfExperts: {
        constructor: SwarmOfExperts,
        prefix: 'moe'
    },
    Autoencoder: {
        constructor: Autoencoder,
        prefix: 'dia'
    },
    FastAssociativeMemory: {
        constructor: FastAssociativeMemory,
        prefix: 'mem'
    },
    CapsNet: {
        constructor: CapsNet,
        prefix: 'cap'
    },
    GPT2Attention: {
        constructor: GPT2Attention,
        prefix: 'attn'
    },
    Range: {
        constructor: Range,
        prefix: 'op'
    },
    SinusoidalPositionalEncoding: {
        constructor: SinusoidalPositionalEncoding,
        prefix: 'enc'
    },
    SynthesizerAttention: {
        constructor: SynthesizerAttention,
        prefix: 'attn'
    },
    LambdaLayer: {
        constructor: LambdaLayer,
        prefix: 'op'
    },
    ProjectedFeatureReduction: {
        constructor: ProjectedFeatureReduction,
        prefix: 'op'
    },
    RotaryPositionalEncoding: {
        constructor: RotaryPositionalEncoding,
        prefix: 'enc'
    },
    LinearAttention: {
        constructor: LinearAttention,
        prefix: 'attn'
    },
    Antirectifier: {
        constructor: Antirectifier,
        prefix: 'op'
    },
    SqueezeAndExcitation: {
        constructor: SqueezeAndExcitation,
        prefix: 'op'
    },
    EfficientChannelAttention: {
        constructor: EfficientChannelAttention,
        prefix: 'attn'
    },
    FourierFeaturePositionalEncoding: {
        constructor: FourierFeaturePositionalEncoding,
        prefix: 'enc'
    },
    IncrementalPowerIterationPCA: {
        constructor: IncrementalPowerIterationPCA,
        prefix: 'pca'
    },
    AttentionFreeTransformer: {
        constructor: AttentionFreeTransformer,
        prefix: 'attn'
    }
}

/**
 * @type {{[K in keyof typeof customLayersConfig]: (config: any) => import('@tensorflow/tfjs').layers.Layer}}
 */
const customLayers = Object.fromEntries(
    Object.entries(customLayersConfig).map(([key, { constructor, prefix }]) => [
        key,
        createLayerFactory(constructor, prefix)
    ])
)

export default customLayers

class StateSpace extends LayerBase {
    constructor(config) {
        super({ name: `ssm-${randomString()}`, ...config })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.decayFactor = config.decayFactor || 1.0
        this.activation = config.activation || 'tanh'
        this.beta = config.beta || 1.0
    }

    build(inputShape) {
        const inputDim = inputShape[2]
        this.kernel = this.addWeight(
            'kernel',
            [inputDim, this.innerDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.recurrentKernel = this.addWeight(
            'recurrentKernel',
            [this.units, this.innerDim],
            'float32',
            tf.initializers.orthogonal({ gain: 1 })
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.units, this.units], // Change this line
            'float32',
            tf.initializers.glorotNormal()
        )
        this.bias = this.addWeight(
            'bias',
            [this.innerDim],
            'float32',
            tf.initializers.zeros()
        )
        this.meanKernel = this.addWeight(
            'meanKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.meanBias = this.addWeight(
            'meanBias',
            [this.units],
            'float32',
            tf.initializers.zeros()
        )
        this.logVarKernel = this.addWeight(
            'logVarKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.logVarBias = this.addWeight(
            'logVarBias',
            [this.units],
            'float32',
            tf.initializers.zeros()
        )
        this.residual = new ResidualConnection()
    }

    sampleLatentState(innerState, kwargs) {
        return tf.tidy(() => {
            const mean = tf.add(
                tf.matMul(innerState, this.meanKernel.read()),
                this.meanBias.read()
            )
            const logVar = tf.add(
                tf.matMul(innerState, this.logVarKernel.read()),
                this.logVarBias.read()
            )
            const expLogVar = logVar.exp()

            if (kwargs.training) {
                // Compute the KL Divergence
                const klDivergence = logVar
                    .add(1)
                    .sub(mean.square())
                    .sub(expLogVar)
                    .mean()
                    .mul(-0.5)
                    .mul(this.beta)

                // Add it to the loss function
                if (!this.extraLoss) this.extraLoss = tf.keep(klDivergence)
                else {
                    const oldValue = this.extraLoss
                    this.extraLoss = tf.keep(this.extraLoss.add(klDivergence))
                    oldValue.dispose()
                }
            }

            // Sample from the latent space using the reparameterization trick
            const epsilon = tf.randomNormal(mean.shape)
            return tf.add(mean, tf.mul(epsilon, expLogVar.sqrt()))
        })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, sequenceLength, inputDim] = inputs.shape

            let state = tf.zeros([batchSize, this.units])
            const outputs = []

            const kernel = this.kernel.read()
            const recurrentKernel = this.recurrentKernel.read()
            const outputKernel = this.outputKernel.read()
            const bias = this.bias.read()

            for (let t = 0; t < sequenceLength; t++) {
                const input = inputs
                    .slice([0, t, 0], [batchSize, 1, inputDim])
                    .reshape([batchSize, inputDim])
                const innerState = tf
                    .add(
                        tf.matMul(input, kernel),
                        tf.matMul(state, recurrentKernel).mul(this.decayFactor)
                    )
                    .add(bias)
                const activatedState = tf.layers
                    .activation({ activation: this.activation })
                    .apply(innerState)
                const latentState = tf.tidy(() => {
                    return this.sampleLatentState(activatedState, kwargs)
                })
                const newState = tf.matMul(latentState, outputKernel)
                outputs.push(newState)
                state = newState
            }

            let output = this.returnSequences
                ? tf.stack(outputs, 1)
                : outputs[outputs.length - 1]

            output = this.rmsNorm(output)

            return this.residual.apply([inputs, output])
        })
    }

    computeOutputShape(inputShape) {
        const outputShape = this.returnSequences
            ? [inputShape[0], inputShape[1], this.units]
            : [inputShape[0], this.units]
        return outputShape
    }

    getWeights() {
        return [
            this.kernel.read(),
            this.outputKernel.read(),
            this.recurrentKernel.read(),
            this.bias.read(),
            this.meanKernel.read(),
            this.meanBias.read(),
            this.logVarKernel.read(),
            this.logVarBias.read()
        ]
    }

    setWeights(weights) {
        this.kernel.write(weights[0])
        this.outputKernel.write(weights[1])
        this.recurrentKernel.write(weights[2])
        this.bias.write(weights[3])
        this.meanKernel.write(weights[4])
        this.meanBias.write(weights[5])
        this.logVarKernel.write(weights[6])
        this.logVarBias.write(weights[7])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            decayFactor: this.decayFactor,
            activation: this.activation,
            beta: this.beta
        }
    }
}

class ChunkedStateSpace extends StateSpace {
    constructor(config) {
        super({ name: `css-${randomString()}`, ...config })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.epsilon = config.epsilon || false
        this.chunkSize = config.chunkSize || 4
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, sequenceLength, inputDim] = inputs.shape

            let state = tf.zeros([batchSize, this.units])
            const outputs = []

            const kernel = this.kernel.read()
            const recurrentKernel = this.recurrentKernel.read()
            const outputKernel = this.outputKernel.read()
            const bias = this.bias.read()

            const numChunks = Math.ceil(sequenceLength / this.chunkSize)

            for (let c = 0; c < numChunks; c++) {
                const chunkStart = c * this.chunkSize
                const chunkEnd = Math.min(
                    chunkStart + this.chunkSize,
                    sequenceLength
                )
                const chunkLength = chunkEnd - chunkStart

                const inputChunk = inputs
                    .slice(
                        [0, chunkStart, 0],
                        [batchSize, chunkLength, inputDim]
                    )
                    .reshape([batchSize * chunkLength, inputDim])

                const innerStateChunk = tf.tanh(
                    tf.add(
                        tf.add(
                            tf.matMul(inputChunk, kernel),
                            tf.matMul(
                                tf.tile(state, [chunkLength, 1]),
                                recurrentKernel
                            )
                        ),
                        bias
                    )
                )

                const newStateChunk = tf.matMul(innerStateChunk, outputKernel)

                state = newStateChunk.slice(
                    [batchSize * (chunkLength - 1), 0],
                    [batchSize, this.units]
                )

                outputs.push(
                    newStateChunk.reshape([batchSize, chunkLength, this.units])
                )
            }

            let output = this.returnSequences
                ? tf.concat(outputs, 1)
                : outputs[outputs.length - 1]

            if (this.layernorm) output = this.layernorm.apply(output)

            return this.residual.apply([inputs, output])
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            chunkSize: this.chunkSize
        }
    }
}

const exportedLayers = [
    // IndependentComponentAnalysis,
    // AdaptiveMixtureOfExperts,
    // Antirectifier,
    // AttentionFreeTransformer,
    // Autoencoder,
    // Bias,
    // CapsNet,
    // CausalSelfAttention,
    ChunkedStateSpace,
    // DeterministicEmbedding,
    // EfficientAttention,
    // EfficientChannelAttention,
    // FastAssociativeMemory,
    // FourierFeaturePositionalEncoding,
    // GatedLinearMLP,
    // GroupedQueryAttention,
    // IncrementalPowerIterationPCA,
    // LambdaLayer,
    // LinearAttention,
    // RandomFeatureAttention,
    // RandomProjectionFeatureReduction,
    // LocalSelfAttention,
    // VariableDimensionMLP,
    // MixtureOfDepths,
    // MixtureOfExperts,
    // MultiHeadAttention,
    // // MultiHeadMoeBlock,
    // // MultiLayerPerceptron,
    // MultiQueryAttention,
    // Range,
    // ResidualConnection,
    // RotaryPositionalEncoding,
    // SelfAttention,
    // SharedEmbedding,
    // SinusoidalPositionalEncoding,
    // SMEARMoE,
    // SparseMixtureOfExperts,
    // SqueezeAndExcitation,
    StateSpace
    // SwarmOfExperts,
    // SynthesizerAttention,
    // WeightedSum
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})
