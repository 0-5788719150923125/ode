import * as tf from '@tensorflow/tfjs'
import customOps from './ops.js'
import customActivations from './activations.js'
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
//     activation: (config) =>
//         tf.layers.activation({ name: `act-${randomString()}`, ...config }),
//     add: (config) =>
//         tf.layers.add({ name: `add-${randomString()}`, ...config }),
//     bottleneck: (config) =>
//         tf.layers.dense({ name: `bot-${randomString()}`, ...config }),
//     concatenate: (config) =>
//         tf.layers.concatenate({ name: `con-${randomString()}`, ...config }),
//     conv1d: (config) =>
//         tf.layers.conv1d({ name: `c1d-${randomString()}`, ...config }),
//     conv2d: (config) =>
//         tf.layers.conv2d({ name: `c2d-${randomString()}`, ...config }),
//     dense: (config) =>
//         tf.layers.dense({ name: `ffd-${randomString()}`, ...config }),
//     embedding: (config) =>
//         tf.layers.embedding({ name: `emb-${randomString()}`, ...config }),
//     input: (config) =>
//         tf.layers.input({ name: `inp-${randomString()}`, ...config }),
//     multiply: (config) =>
//         tf.layers.multiply({ name: `mul-${randomString()}`, ...config }),
//     timeDistributed: (config) =>
//         tf.layers.timeDistributed({
//             name: `time-${randomString()}`,
//             ...config
//         }),
//     gru: (config) =>
//         tf.layers.gru({ name: `gru-${randomString()}`, ...config }),
//     lstm: (config) =>
//         tf.layers.lstm({ name: `lstm-${randomString()}`, ...config }),
//     rnn: (config) =>
//         tf.layers.gru({ name: `rnn-${randomString()}`, ...config }),
//     SharedEmbedding: (config) =>
//         new SharedEmbedding({ name: `emb-${randomString()}`, ...config }),
//     RandomFeatureAttention: (config) =>
//         new RandomFeatureAttention({
//             name: `attn-${randomString()}`,
//             ...config
//         }),
//     SMEAR: (config) =>
//         new SMEAR({
//             name: `moe-${randomString()}`,
//             ...config
//         }),
//     MultiLayerPerceptron: (config) =>
//         new MultiLayerPerceptron({
//             name: `mlp-${randomString()}`,
//             ...config
//         }),
//     GatedLinearMLP: (config) =>
//         new GatedLinearMLP({
//             name: `mlp-${randomString()}`,
//             ...config
//         })
// }
// export default customLayers

class LayerBase extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.ode = {
            ops: customOps
        }
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            // stuff should go here
        })
    }

    // Direct application of matMul to x and kernel throws:
    // > Error in gradient for op BatchMatMul.
    // > The gradient of input 'b' has shape '16,48,48',
    // > which does not match the shape of the input '48,48'
    // Two solutions worked:
    // 1. Use tf.layers.dense but reassign kernel and bias
    // 2. Use tf.matMul but expandDims and tile kernel (current)
    // Another option, of course, is to separate attention logic
    // from trainable weights completely and use tf.layers.dense
    // inside a model definition. I was not able to define fully
    // function regular dense layers inside a custom layer.
    // Something related to how weights are loaded with this.kernel
    // and duplicating names

    applyDense(x, kernel, bias) {
        let k = kernel.expandDims(0).tile([x.shape[0], 1, 1])
        const m = tf.matMul(x, k)
        if (bias) return tf.add(m, bias)
        else return m
    }

    rmsNorm = (x) => {
        const rms = tf.sqrt(tf.mean(tf.square(x), -1, true))
        const epsilon = 1e-7
        return x.div(rms.add(epsilon))
    }

    // findLayer(key) {
    //     const lowercaseKey = key.toLowerCase()
    //     const match = Object.keys(customLayers).find(
    //         (k) => k.toLowerCase() === lowercaseKey
    //     )
    //     return match ? customLayers[match] : undefined
    // }

    applyALiBi(scores, numHeads, currentHead, seqLen) {
        if (!this.alibiSlope) {
            const base = tf.scalar(2 ** 8)
            const powers = tf
                .range(0, numHeads)
                .cast('float32')
                .add(tf.scalar(1))
            const slopes = tf.pow(base, powers.div(tf.scalar(numHeads)))
            this.alibiSlope = tf.keep(
                slopes.reciprocal().expandDims(-1).expandDims(-1)
            )
        }
        const alibiSlope = this.alibiSlope.gather([currentHead])
        const range = tf.range(0, seqLen)
        const relativePositions = range.expandDims(1).sub(range.expandDims(0))
        const alibiScores = tf.mul(alibiSlope, relativePositions)

        const adjustedAlibiScores = alibiScores.slice(
            [0, 0, 0],
            [1, seqLen, scores.shape[2]]
        )
        const expandedAlibiScores = adjustedAlibiScores.tile([
            scores.shape[0],
            1,
            1
        ])

        return scores.add(expandedAlibiScores)
    }

    static get className() {
        return this.name
    }

    getConfig() {
        return {
            ...super.getConfig(),
            className: this.getClassName()
        }
    }
}

class ResidualConnection extends LayerBase {
    constructor(config) {
        super(config)
    }

    computeOutputShape(inputShape) {
        // inputShape[0] and inputShape[1 should be identical
        return inputShape[0]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            // inputs is an array where inputs[0] is the original input and inputs[1] is the output to be added to it.
            if (inputs.length !== 2) {
                throw new Error('ResidualConnection expects 2 inputs.')
            }

            const [originalInput, blockOutput] = inputs
            return tf.add(originalInput, blockOutput)
        })
    }
}

class LinearAttention extends LayerBase {
    constructor(config) {
        super({ name: `attn-${randomString()}`, ...config })
        this.units = config.units || 64
        this.projection = config.projection || 256
        this.numFeatures = config.numFeatures || 256
    }

    build(inputShape) {
        this.query = tf.layers.dense({
            units: this.projection,
            activation: 'linear',
            useBias: false,
            kernelInitializer: tf.initializers.glorotUniform()
        })
        this.key = tf.layers.dense({
            units: this.projection,
            activation: 'linear',
            useBias: false,
            kernelInitializer: tf.initializers.glorotUniform()
        })
        this.value = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            useBias: false,
            kernelInitializer: tf.initializers.glorotUniform()
        })
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.query.apply(inputs)
            const K = this.key.apply(inputs)
            const V = this.value.apply(inputs)

            const Qp = this.generateRandomFeatures(Q)
            const Kp = this.generateRandomFeatures(K)

            const scores = tf.matMul(Qp, Kp, false, true)

            const weights = scores.div(tf.scalar(this.numFeatures))

            const outputs = tf.matMul(weights, V)

            return this.residual.apply([inputs, outputs])
        })
    }

    generateRandomFeatures(inputs) {
        const dims = inputs.shape[inputs.shape.length - 1]
        const W = tf.randomNormal([dims, this.numFeatures])
        const b = tf.randomUniform([this.numFeatures], 0, 2 * Math.PI)
        const features = tf.matMul(inputs, W).add(b).cos()
        return features
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            projection: this.projection,
            numFeatures: this.numFeatures
        }
    }
}

class Antirectifier extends LayerBase {
    constructor(config) {
        super({ name: `anti-${randomString()}`, ...config })
        // TODO(bileschi): Can we point to documentation on masking here?
        this.supportsMasking = true
    }

    /**
     * This layer only works on 4D Tensors [batch, height, width, channels],
     * and produces output with twice as many channels.
     *
     * layer.computeOutputShapes must be overridden in the case that the output
     * shape is not the same as the input shape.
     * @param {*} inputShapes
     */
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], inputShape[2], 2 * inputShape[3]]
    }

    /**
     * Centers the input and applies the following function to every element of
     * the input.
     *
     *     x => [max(x, 0), max(-x, 0)]
     *
     * The theory being that there may be signal in the both negative and positive
     * portions of the input.  Note that this will double the number of channels.
     * @param inputs Tensor to be treated.
     * @param kwargs Only used as a pass through to call hooks.  Unused in this
     *   example code.
     */
    call(inputs, kwargs) {
        let input = inputs
        if (Array.isArray(input)) {
            input = input[0]
        }

        const origShape = input.shape
        const flatShape = [
            origShape[0],
            origShape[1] * origShape[2] * origShape[3]
        ]
        const flattened = input.reshape(flatShape)
        const centered = tf.sub(flattened, flattened.mean(1).expandDims(1))
        const pos = centered.relu().reshape(origShape)
        const neg = centered.neg().relu().reshape(origShape)
        return tf.concat([pos, neg], 3)
    }
}

class RotaryPositionalEncoding extends LayerBase {
    constructor(config) {
        super({ name: `rot-${randomString()}`, ...config })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const seqLen = inputs.shape[1]
            const embeddingDim = inputs.shape[2]
            const posEncoding = this.getRotaryPositionalEmbedding(
                seqLen,
                embeddingDim
            )
            const output = this.applyRotaryPositionalEmbedding(
                inputs,
                posEncoding
            )
            return output.slice([0, 0, 0], inputs.shape)
        })
    }

    getRotaryPositionalEmbedding(seqLen, embeddingDim) {
        const pos = tf.range(0, seqLen, 1, 'float32').reshape([-1, 1])
        const i = tf.range(0, embeddingDim / 2, 1, 'float32').reshape([1, -1])
        const angleRates = tf.pow(10000, tf.div(i, embeddingDim / 2))
        const angleRads = tf.mul(pos, tf.div(1, angleRates))
        const sin = tf.sin(angleRads)
        const cos = tf.cos(angleRads)

        // Interleave sin and cos values
        const sinExpanded = sin.expandDims(2) // Expanding dimension to enable interleaving
        const cosExpanded = cos.expandDims(2)
        const concatenated = tf.concat([sinExpanded, cosExpanded], 2) // Concatenate along the new axis
        const posEncoding = concatenated.reshape([seqLen, embeddingDim])
        return posEncoding
    }

    applyRotaryPositionalEmbedding(x, posEncoding) {
        const embeddingDim = x.shape[2]
        const xDtype = x.dtype

        // Split the embedding dimension into two halves for sin and cos applications
        const rotaryDim = embeddingDim / 2
        const [xRot, xPass] = tf.split(x, 2, -1)

        // Apply sin to the first half and cos to the second half of posEncoding
        const sinEncoding = posEncoding.slice([0, 0], [-1, rotaryDim])
        const cosEncoding = posEncoding.slice([0, rotaryDim], [-1, -1])

        // Apply the encodings
        const xRotSin = tf.mul(xRot, sinEncoding)
        const xRotCos = tf.mul(xRot, cosEncoding)

        // Reconstruct the rotated embeddings
        const rotatedX = tf.concat([xRotSin, xRotCos], -1)

        // Concatenate the rotated part with the part that does not get rotated
        const output = tf.concat([rotatedX, xPass], -1)

        return output.asType(xDtype)
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig()
        }
    }
}

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

// https://arxiv.org/abs/1709.01507
class SqueezeAndExcitation extends LayerBase {
    constructor(config) {
        super({ name: `se-${randomString()}`, ...config })
        this.ratio = config.ratio
    }

    build(inputShape) {
        this.units = inputShape[inputShape.length - 1]

        this.squeeze = tf.layers.globalAveragePooling1d()

        this.exciteDense1 = customLayers.dense({
            units: Math.max(1, Math.floor(this.units / this.ratio)),
            activation: 'relu',
            kernelInitializer: 'heNormal',
            useBias: false
        })

        this.exciteDense2 = customLayers.dense({
            units: this.units,
            activation: 'sigmoid',
            kernelInitializer: 'heNormal',
            useBias: false
        })
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const squeezed = this.squeeze.apply(inputs)
            const excited = this.exciteDense1.apply(squeezed)
            const excitedOutput = this.exciteDense2.apply(excited)
            return inputs.mul(excitedOutput.expandDims(-2))
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            ratio: this.ratio
        }
    }
}

class EfficientChannelAttention extends LayerBase {
    constructor(config) {
        super({ name: `eca-${randomString()}`, ...config })
        this.gamma = config.gamma || 2
    }

    build(inputShape) {
        this.channels = inputShape[inputShape.length - 1]
        this.kernelSize = Math.max(1, Math.floor(this.channels / this.gamma))

        this.conv1d = customLayers.conv1d({
            filters: 1,
            kernelSize: this.kernelSize,
            strides: 1,
            padding: 'same',
            activation: 'sigmoid',
            kernelInitializer: 'ones',
            useBias: false
        })
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const avgPool = tf.mean(inputs, [1], true)
            const attention = this.conv1d.apply(avgPool)

            return inputs.mul(attention)
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            gamma: this.gamma
        }
    }
}

class FourierFeaturePositionalEncoding extends LayerBase {
    constructor(config) {
        super(config)
        this.numFeatures = config.numFeatures
        this.scale = config.scale || 1.0
    }

    call(inputs) {
        const position = tf.range(0, inputs.shape[1], 1, 'float32')
        const angles = tf.mul(
            position,
            tf.pow(10000.0, tf.linspace(0.0, 1.0, this.numFeatures))
        )
        const cosFeatures = tf.cos(angles)
        const sinFeatures = tf.sin(angles)
        const features = tf.stack([cosFeatures, sinFeatures], -1)
        const flattened = tf.reshape(features, [1, -1, this.numFeatures * 2])
        const scaled = tf.mul(flattened, this.scale)
        return tf.add(inputs, scaled)
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            numFeatures: this.numFeatures,
            scale: this.scale
        })
        return config
    }
}

class Bias extends LayerBase {
    constructor(config) {
        super({ name: `bias-${randomString()}`, ...config })
        this.l1 = config.l1 || 0
        this.l2 = config.l2 || 0
    }

    build(inputShape) {
        const biasShape = inputShape[inputShape.length - 1]
        this.bias = this.addWeight(
            'bias',
            [biasShape],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l1l2({
                l1: this.l1,
                l2: this.l2
            })
        )
    }

    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        return tf.add(inputs, this.bias.read())
    }

    getConfig() {
        return { ...super.getConfig() }
    }
}

class WeightedSum extends LayerBase {
    constructor(config) {
        super({ name: `wsum-${randomString()}`, ...config })
        this.units = config.units || 1
    }

    build(inputShape) {
        if (!Array.isArray(inputShape) || inputShape.length < 2) {
            throw new Error('WeightedSum layer expects at least two inputs.')
        }

        const numInputs = inputShape.length
        this.kernel = []

        for (let i = 0; i < numInputs; i++) {
            this.kernel.push(
                this.addWeight(
                    `weight_${i}`,
                    [inputShape[i][inputShape[i].length - 1], this.units],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
        }
    }

    call(inputs) {
        return tf.tidy(() => {
            const weightedInputs = []

            for (let i = 0; i < inputs.length; i++) {
                const weightedInput = this.applyDense(inputs[i], this.kernel[i])
                weightedInputs.push(weightedInput)
            }

            const output = weightedInputs.reduce((sum, input) => sum.add(input))
            return output
        })
    }

    dense(x, kernel) {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
    }

    computeOutputShape(inputShape) {
        return [inputShape[0][0], this.units]
    }

    getWeights() {
        return this.kernel.map((weight) => weight.read())
    }

    setWeights(weights) {
        for (let i = 0; i < weights.length; i++) {
            this.kernel[i].write(weights[i])
        }
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units
        }
    }
}

// https://github.com/rish-16/aft-pytorch/blob/main/aft_pytorch/aft_pytorch.py
class AttentionFreeTransformer extends LayerBase {
    constructor(config) {
        super({ name: `aft-${randomString()}`, ...config })
        this.units = config.units || 64
        this.hiddenDim = config.hiddenDim || 64
        this.contextLength = config.contextLength
    }

    build(inputShape) {
        this.toQ = this.addWeight(
            'toQ',
            [inputShape[inputShape.length - 1], this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.toK = this.addWeight(
            'toK',
            [inputShape[inputShape.length - 1], this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.toV = this.addWeight(
            'toV',
            [inputShape[inputShape.length - 1], this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.project = this.addWeight(
            'project',
            [this.hiddenDim, this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.wbias = this.addWeight(
            'wbias',
            [this.contextLength, this.contextLength],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [B, T, _] = inputs.shape

            const Q = this.applyDense(inputs, this.toQ.read()).reshape([
                B,
                T,
                this.hiddenDim
            ])
            const K = this.applyDense(inputs, this.toK.read()).reshape([
                B,
                T,
                this.hiddenDim
            ])
            const V = this.applyDense(inputs, this.to.read()).reshape([
                B,
                T,
                this.hiddenDim
            ])

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const tempWbias = this.wbias
                .read()
                .slice([0, 0], [T, T])
                .expandDims(0)
                .add(mask)

            const QSig = tf.sigmoid(Q)
            const temp = tf.matMul(tf.exp(tempWbias), tf.mul(tf.exp(K), V))
            const weighted = tf.div(
                temp,
                tf.matMul(tf.exp(tempWbias), tf.exp(K))
            )

            const Yt = tf.mul(QSig, weighted)

            const outputs = this.applyDense(
                Yt.reshape([B, T, this.hiddenDim]),
                this.project.read()
            )

            return this.residual.apply([inputs, outputs])
        })
    }

    getWeights() {
        return [
            this.toQ.read(),
            this.toK.read(),
            this.toV.read(),
            this.project.read(),
            this.wbias.read()
        ]
    }

    setWeights(weights) {
        this.toQ.write(weights[0])
        this.toK.write(weights[1])
        this.toV.write(weights[2])
        this.project.write(weights[3])
        this.wbias.write(weights[4])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            hiddenDim: this.hiddenDim,
            contextLength: this.contextLength
        }
    }
}

class IncrementalPowerIterationPCA extends LayerBase {
    constructor(config) {
        super({ name: `pca-${randomString()}`, ...config })
        this.outputDim = config.outputDim
        this.epsilon = config.epsilon || 1e-7
        this.numIterations = config.numIterations || 10
        this.mean = null
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.inputDim = inputDim
        this.components = this.addWeight(
            'components',
            [this.inputDim, this.outputDim],
            'float32',
            tf.initializers.glorotNormal({})
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            this.fit(inputs)

            // Center the data
            const centered = tf.sub(inputs, this.mean)

            // Project data onto principal components
            const flattenedCentered = tf.reshape(centered, [-1, this.inputDim])
            const components = this.components.read()

            const result = tf.matMul(flattenedCentered, components)
            const reshapedResult = tf.reshape(result, [
                ...inputs.shape.slice(0, -1),
                this.outputDim
            ])

            return reshapedResult
        })
    }

    fit(X) {
        tf.tidy(() => {
            const flattenedX = tf.reshape(X, [-1, this.inputDim])

            // Compute mean
            this.mean = tf.mean(flattenedX, 0, true)

            // Center the data
            const centered = tf.sub(flattenedX, this.mean)

            // Compute covariance matrix
            const cov = tf
                .matMul(centered, centered, true, false)
                .div(tf.scalar(centered.shape[0] - 1))

            // Compute principal components using power iteration
            const components = this.powerIteration(cov)
        })
    }

    powerIteration(cov) {
        return tf.tidy(() => {
            const [n] = cov.shape
            let components = tf.randomNormal([n, this.outputDim])

            for (let iter = 0; iter < this.numIterations; iter++) {
                // Power iteration
                components = tf.matMul(cov, components)

                // Orthogonalize
                components = tf.linalg.gramSchmidt(components)
            }

            return components
        })
    }

    computeOutputShape(inputShape) {
        return [...inputShape.slice(0, -1), this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim,
            epsilon: this.epsilon,
            numIterations: this.numIterations
        }
    }
}

class RandomProjectionFeatureReduction extends LayerBase {
    constructor(config) {
        super({ name: `proj-${randomString()}`, ...config })
        this.outputDim = config.outputDim
        this.scale = config.scale || 1.0
        this.seed = config.seed || 42
    }

    build(inputShape) {
        this.inputDim = inputShape[inputShape.length - 1]
        this.projectionMatrix = tf.randomNormal(
            [this.inputDim, this.outputDim],
            0,
            this.scale / Math.sqrt(this.outputDim),
            'float32',
            this.seed
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLength, inputDim] = input.shape

            // Reshape to 2D for matrix multiplication
            const inputReshaped = input.reshape([-1, inputDim])

            // Apply random projection
            const projected = tf.matMul(inputReshaped, this.projectionMatrix)

            // Reshape back to 3D
            return projected.reshape([batchSize, seqLength, this.outputDim])
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim,
            scale: this.scale,
            seed: this.seed
        }
    }
}

const exportedLayers = [
    // IndependentComponentAnalysis,
    // AdaptiveMixtureOfExperts,
    Antirectifier,
    AttentionFreeTransformer,
    // Autoencoder,
    Bias,
    // CapsNet,
    // CausalSelfAttention,
    ChunkedStateSpace,
    // DeterministicEmbedding,
    // EfficientAttention,
    EfficientChannelAttention,
    // FastAssociativeMemory,
    FourierFeaturePositionalEncoding,
    // GatedLinearMLP,
    // GroupedQueryAttention,
    IncrementalPowerIterationPCA,
    // LambdaLayer,
    LinearAttention,
    // RandomFeatureAttention,
    RandomProjectionFeatureReduction,
    // LocalSelfAttention,
    // VariableDimensionMLP,
    // MixtureOfDepths,
    // MixtureOfExperts,
    // MultiHeadAttention,
    // // MultiHeadMoeBlock,
    // // MultiLayerPerceptron,
    // MultiQueryAttention,
    // Range,
    ResidualConnection,
    RotaryPositionalEncoding,
    // SelfAttention,
    // SharedEmbedding,
    // SinusoidalPositionalEncoding,
    // SMEARMoE,
    // SparseMixtureOfExperts,
    SqueezeAndExcitation,
    StateSpace,
    // SwarmOfExperts,
    // SynthesizerAttention,
    WeightedSum
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})
