import * as tf from '@tensorflow/tfjs'
import customOps from './ops.js'
import customActivations from './activations.js'
import Expert from './experts.js'
import {
    randomString,
    seededPRNG,
    seededValueFromArray,
    shuffleArray
} from './utils.js'

const customLayers = {
    activation: (config) =>
        tf.layers.activation({ name: `act-${randomString()}`, ...config }),
    add: (config) =>
        tf.layers.add({ name: `add-${randomString()}`, ...config }),
    bottleneck: (config) =>
        tf.layers.dense({ name: `bot-${randomString()}`, ...config }),
    concatenate: (config) =>
        tf.layers.concatenate({ name: `con-${randomString()}`, ...config }),
    conv1d: (config) =>
        tf.layers.conv1d({ name: `c1d-${randomString()}`, ...config }),
    conv2d: (config) =>
        tf.layers.conv2d({ name: `c2d-${randomString()}`, ...config }),
    dense: (config) =>
        tf.layers.dense({ name: `ffd-${randomString()}`, ...config }),
    embedding: (config) =>
        tf.layers.embedding({ name: `emb-${randomString()}`, ...config }),
    input: (config) =>
        tf.layers.input({ name: `inp-${randomString()}`, ...config }),
    multiply: (config) =>
        tf.layers.multiply({ name: `mul-${randomString()}`, ...config }),
    timeDistributed: (config) =>
        tf.layers.timeDistributed({
            name: `time-${randomString()}`,
            ...config
        }),
    gru: (config) =>
        tf.layers.gru({ name: `gru-${randomString()}`, ...config }),
    lstm: (config) =>
        tf.layers.lstm({ name: `lstm-${randomString()}`, ...config }),
    rnn: (config) => tf.layers.gru({ name: `rnn-${randomString()}`, ...config })
}
export default customLayers

class LayerBase extends tf.layers.Layer {
    constructor(config) {
        super(config)
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
        if (bias) {
            return tf.add(m, bias)
        } else {
            return m
        }
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

class SharedEmbedding extends LayerBase {
    constructor(config) {
        super({ name: `emb-${randomString()}`, ...config })
        this.inputDim = config.inputDim
        this.outputDim = config.outputDim
        this.embeddingsInitializer =
            config.embeddingsInitializer || 'glorotUniform'
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        this.embeddings = this.addWeight(
            'embeddings',
            [this.inputDim, this.outputDim],
            'float32',
            tf.initializers[this.embeddingsInitializer](),
            tf.regularizers.l2({ l2: 0.1 })
        )
    }

    computeOutputShape(inputShape) {
        if (inputShape.length === 2) {
            // Input embedding
            return [inputShape[0], inputShape[1], this.outputDim]
        } else if (inputShape.length === 3) {
            // Output projection
            return [inputShape[0], inputShape[1], this.inputDim]
        } else {
            throw new Error('Invalid input shape for TiedEmbedding layer.')
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            if (inputs.shape.length === 2) {
                // Input embedding
                const flatInputs = tf.reshape(inputs, [-1])
                const embeddings = tf.gather(
                    this.embeddings.read(),
                    flatInputs.cast('int32')
                )

                let outputs = tf.reshape(embeddings, [
                    inputs.shape[0],
                    inputs.shape[1],
                    this.outputDim
                ])

                outputs = kwargs['training']
                    ? tf.dropout(outputs, this.dropout)
                    : outputs

                return outputs
            } else if (inputs.shape.length === 3) {
                // Output projection
                const denseOutput = tf.matMul(
                    tf.reshape(inputs, [-1, this.outputDim]),
                    this.embeddings.read(),
                    false,
                    true
                )

                let outputs = tf.reshape(denseOutput, [
                    inputs.shape[0],
                    inputs.shape[1],
                    this.inputDim
                ])

                outputs = kwargs['training']
                    ? tf.dropout(outputs, this.dropout)
                    : outputs

                return outputs
            } else {
                throw new Error(
                    'Invalid input shape for SharedEmbedding layer.'
                )
            }
        })
    }

    getWeights() {
        return [this.embeddings.read()]
    }

    setWeights(weights) {
        this.embeddings.write(weights[0])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            inputDim: this.inputDim,
            outputDim: this.outputDim
        }
    }
}

class SelfAttention extends LayerBase {
    constructor(config) {
        super({ name: `attn-${randomString()}`, ...config })
        this.projection = config.projection || 256
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernel = this.addWeight(
            `queryKernel`,
            [inputDim, this.projection],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.keyKernel = this.addWeight(
            `keyKernel`,
            [inputDim, this.projection],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.valueKernel = this.addWeight(
            `valueKernel`,
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.applyDense(inputs, this.queryKernel.read())
            const K = this.applyDense(inputs, this.keyKernel.read())
            const V = this.applyDense(inputs, this.valueKernel.read())

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf
                .matMul(Q, K, false, true)
                .div(tf.scalar(this.projection).sqrt())
                .add(mask)

            const weights = scores.softmax()

            let outputs = tf.matMul(weights, V)

            outputs = this.rmsNorm(outputs)

            return this.residual.apply([inputs, outputs])
        })
    }

    getWeights() {
        return [
            this.queryKernel.read(),
            this.keyKernel.read(),
            this.valueKernel.read()
        ]
    }

    setWeights(weights) {
        this.queryKernel.write(weights[0])
        this.keyKernel.write(weights[1])
        this.valueKernel.write(weights[2])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            projection: this.projection
        }
    }
}

// https://github.com/cmsflash/efficient-attention/blob/master/efficient_attention.py
// currently failing, because the causal mask makes loss values extremely high
class EfficientAttention extends LayerBase {
    constructor(config) {
        super({ name: `attn-${randomString()}`, ...config })
        this.keyChannels = config.keyChannels || 256
        this.valueChannels = config.valueChannels || 256
        this.headCount = config.headCount || 8
        this.contextLength = config.contextLength
    }

    build(inputShape) {
        const inputDepth = inputShape[inputShape.length - 1]

        this.queries = this.addWeight(
            'queries',
            [1, inputDepth, this.keyChannels],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.keys = this.addWeight(
            'keys',
            [1, inputDepth, this.keyChannels],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.values = this.addWeight(
            'values',
            [1, inputDepth, this.valueChannels],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.reprojection = this.addWeight(
            'reprojection',
            [1, this.valueChannels, inputDepth],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLen, dims] = inputs.shape

            const queries = tf
                .conv1d(
                    inputs.reshape([batchSize, seqLen, dims]),
                    this.queries.read(),
                    1,
                    'same'
                )
                .reshape([batchSize, this.keyChannels, seqLen])
            const keys = tf
                .conv1d(
                    inputs.reshape([batchSize, seqLen, dims]),
                    this.keys.read(),
                    1,
                    'same'
                )
                .reshape([batchSize, this.keyChannels, seqLen])
            const values = tf
                .conv1d(
                    inputs.reshape([batchSize, seqLen, dims]),
                    this.values.read(),
                    1,
                    'same'
                )
                .reshape([batchSize, this.valueChannels, seqLen])

            const headKeyChannels = Math.floor(
                this.keyChannels / this.headCount
            )
            const headValueChannels = Math.floor(
                this.valueChannels / this.headCount
            )

            const mask = tf.linalg
                .bandPart(
                    tf.ones([this.contextLength, this.contextLength]),
                    0,
                    -1
                )
                .sub(tf.eye(this.contextLength))
                .mul(tf.scalar(-1e9))

            const attendedValues = []
            for (let i = 0; i < this.headCount; i++) {
                const key = keys
                    .slice(
                        [0, i * headKeyChannels, 0],
                        [batchSize, headKeyChannels, seqLen]
                    )
                    .softmax(-1)
                const query = queries
                    .slice(
                        [0, i * headKeyChannels, 0],
                        [batchSize, headKeyChannels, seqLen]
                    )
                    .transpose([0, 2, 1])
                    .softmax(-1)
                    .transpose([0, 2, 1])
                const value = values.slice(
                    [0, i * headValueChannels, 0],
                    [batchSize, headValueChannels, seqLen]
                )

                const context = tf.matMul(key, value, false, true).add(mask)
                const attendedValue = tf
                    .matMul(context, query, true, false)
                    .reshape([batchSize, headValueChannels, seqLen])

                attendedValues.push(attendedValue)
            }

            const aggregatedValues = tf
                .concat(attendedValues, 1)
                .reshape([batchSize, seqLen, this.valueChannels])

            const outputs = tf.conv1d(
                aggregatedValues,
                this.reprojection.read(),
                1,
                'same'
            )

            return this.residual.apply([inputs, outputs])
        })
    }

    getWeights() {
        return [
            this.queries.read(),
            this.keys.read(),
            this.values.read(),
            this.reprojection.read()
        ]
    }

    setWeights(weights) {
        this.queries.write(weights[0])
        this.keys.write(weights[1])
        this.values.write(weights[2])
        this.reprojection.write(weights[3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            keyChannels: this.keyChannels,
            valueChannels: this.valueChannels,
            headCount: this.headCount
        }
    }
}

// https://github.com/lhallee/Multi_Head_Mixture_of_Experts__MH-MOE/blob/main/mhmoe.py
class MultiHeadMoeBlock extends LayerBase {
    constructor(config) {
        super({ name: `mh-moe-${randomString()}`, ...config })
        this.hiddenDim = config.hiddenDim || 64
        this.numExperts = config.numExperts || 4
        this.numHeads = config.numHeads || 4
        this.topk = config.topk || 2
        this.headDim = this.hiddenDim / this.numHeads
        this.roundedDim =
            Math.floor(this.hiddenDim / this.numHeads) * this.numHeads
    }

    build(inputShape) {
        this.multiHeadLayer = tf.layers.dense({
            units: this.roundedDim,
            useBias: false,
            activation: 'linear',
            kernelInitializer: 'glorotUniform'
        })

        this.router = new MHRouter({
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            numHeads: this.numHeads
        })

        this.experts = []
        for (let i = 0; i < this.numExperts; i++) {
            const expert = tf.layers.dense({
                units: this.headDim,
                useBias: false,
                activation: 'linear',
                kernelInitializer: 'glorotUniform'
            })
            this.experts.push(expert)
        }

        this.mergeLayer = tf.layers.dense({
            units: this.hiddenDim,
            useBias: false,
            activation: 'linear',
            kernelInitializer: 'glorotUniform'
        })
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const batchSize = inputs.shape[0]
            const seqLen = inputs.shape[1]

            // Project inputs to rounded dimension
            let x = this.multiHeadLayer.apply(inputs)
            x = x.reshape([batchSize * seqLen * this.numHeads, this.headDim])

            // Router
            const routerLogits = this.router.apply(x)
            const routerWeights = routerLogits.softmax()
            const topkOutputs = tf.topk(routerWeights, this.topk)

            // Call experts densely, faster than selective loops
            const expertOutputs = []
            for (const expert of this.experts) {
                expertOutputs.push(expert.apply(x))
            }
            const expertStack = tf.stack(expertOutputs, 1)

            // Select top-k expert outputs
            const batchIndices = tf.range(0, expertStack.shape[0]).expandDims(1)
            const gatherIndices = tf.concat(
                [batchIndices.cast('int32'), topkOutputs.indices.cast('int32')],
                1
            )
            const selectedExpertOutputs = tf.gatherND(
                expertStack.cast('float32'),
                gatherIndices.cast('int32')
            )

            // Multiply selected expert outputs with router weights elementwise
            const weightedExpertOutputs = selectedExpertOutputs.mul(
                topkOutputs.values.expandDims(-1)
            )

            // Combine top-k expert outputs
            x = weightedExpertOutputs.sum(1)

            // Back to original shape
            x = x.reshape([batchSize, seqLen, this.headDim])
            x = this.mergeLayer.apply(x)

            return x
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            numExperts: this.numExperts,
            numHeads: this.numHeads,
            topk: this.topk
        }
    }
}

class MHRouter extends LayerBase {
    constructor(config) {
        super({ name: `mh-router-${randomString()}`, ...config })
        this.numExperts = config.numExperts
        this.hiddenDim = config.hiddenDim
        this.numHeads = config.numHeads
    }

    build(inputShape) {
        this.expertEmbedding = this.addWeight(
            'expertEmbedding',
            [this.hiddenDim / this.numHeads, this.numExperts],
            'float32',
            tf.initializers.randomNormal({ mean: 0, stddev: 1 })
        )
    }

    call(inputs) {
        return tf.matMul(inputs, this.expertEmbedding.read())
    }

    getWeights() {
        return [this.expertEmbedding.read()]
    }

    setWeights(weights) {
        this.expertEmbedding.write(weights[0])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            numHeads: this.numHeads
        }
    }
}

class LocalSelfAttention extends LayerBase {
    constructor(config) {
        super({ name: `local-attn-${randomString()}`, ...config })
        this.units = config.units || 64
        this.projection = config.projection || 256
        this.chunkSize = config.chunkSize || 64
    }

    build(inputShape) {
        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputShape[inputShape.length - 1], this.projection],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputShape[inputShape.length - 1], this.projection],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputShape[inputShape.length - 1], this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.applyDense(inputs, this.queryKernel)
            const K = this.applyDense(inputs, this.keyKernel)
            const V = this.applyDense(inputs, this.valueKernel)

            const numChunks = Math.ceil(inputs.shape[1] / this.chunkSize)
            const chunkOutputs = []

            for (let i = 0; i < numChunks; i++) {
                const start = i * this.chunkSize
                const end = Math.min((i + 1) * this.chunkSize, inputs.shape[1])

                const chunkQ = Q.slice(
                    [0, start, 0],
                    [Q.shape[0], end - start, Q.shape[2]]
                )
                const chunkK = K.slice(
                    [0, start, 0],
                    [K.shape[0], end - start, K.shape[2]]
                )
                const chunkV = V.slice(
                    [0, start, 0],
                    [V.shape[0], end - start, V.shape[2]]
                )

                const scores = tf
                    .matMul(chunkQ, chunkK, false, true)
                    .div(tf.scalar(this.projection).sqrt())

                const weights = scores.softmax()

                const chunkOutput = tf.matMul(weights, chunkV)
                chunkOutputs.push(chunkOutput)
            }

            const outputs = tf.concat(chunkOutputs, 1)

            return this.residual.apply([inputs, outputs])
        })
    }

    applyDense(x, kernel) {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
    }

    getWeights() {
        return [
            this.queryKernel.read(),
            this.keyKernel.read(),
            this.valueKernel.read()
        ]
    }

    setWeights(weights) {
        this.queryKernel.write(weights[0])
        this.keyKernel.write(weights[1])
        this.valueKernel.write(weights[2])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            projection: this.projection,
            chunkSize: this.chunkSize
        }
    }
}

class MultiHeadAttention extends LayerBase {
    constructor(config) {
        super({ name: `mha-${randomString()}`, ...config })
        this.heads = config.heads || 8
        this.projection = config.projection || 64
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.queryBiases = []
        this.keyKernels = []
        this.keyBiases = []
        this.valueKernels = []
        this.valueBiases = []

        for (let i = 0; i < this.heads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.queryBiases.push(
                this.addWeight(
                    `queryBias_${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.keyBiases.push(
                this.addWeight(
                    `keyBias_${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.valueBiases.push(
                this.addWeight(
                    `valueBias_${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.projection * this.heads, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.outputBias = this.addWeight(
            'outputBias',
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.heads; i++) {
                const Q = this.applyDense(
                    inputs,
                    this.queryKernels[i],
                    this.queryBiases[i]
                )
                const K = this.applyDense(
                    inputs,
                    this.keyKernels[i],
                    this.keyBiases[i]
                )
                const V = this.applyDense(
                    inputs,
                    this.valueKernels[i],
                    this.valueBiases[i]
                )

                const scores = tf
                    .matMul(Q, K, false, true)
                    .div(tf.scalar(Math.sqrt(this.projection)))
                    .add(mask)

                let weights = scores.softmax()

                weights = kwargs['training']
                    ? tf.dropout(weights, this.dropout)
                    : weights

                const output = tf.matMul(weights, V)

                attentionOutputs.push(output)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            let outputs = this.applyDense(
                concatenatedOutputs,
                this.outputKernel,
                this.outputBias
            )

            outputs = this.rmsNorm(outputs)

            outputs = this.residual.apply([inputs, outputs])

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getWeights() {
        const weights = []

        for (let i = 0; i < this.heads; i++) {
            weights.push(this.queryKernels[i].read())
            weights.push(this.queryBiases[i].read())
            weights.push(this.keyKernels[i].read())
            weights.push(this.keyBiases[i].read())
            weights.push(this.valueKernels[i].read())
            weights.push(this.valueBiases[i].read())
        }

        weights.push(this.outputKernel.read())
        weights.push(this.outputBias.read())

        return weights
    }

    setWeights(weights) {
        let index = 0

        for (let i = 0; i < this.heads; i++) {
            this.queryKernels[i].write(weights[index++])
            this.queryBiases[i].write(weights[index++])
            this.keyKernels[i].write(weights[index++])
            this.keyBiases[i].write(weights[index++])
            this.valueKernels[i].write(weights[index++])
            this.valueBiases[i].write(weights[index++])
        }

        this.outputKernel.write(weights[index++])
        this.outputBias.write(weights[index])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            heads: this.heads,
            projection: this.projection,
            dropout: this.dropout
        }
    }
}

class MultiQueryAttention extends LayerBase {
    constructor(config) {
        super({ name: `mqa-${randomString()}`, ...config })
        this.projection = config.projection || 256
        this.queries = config.queries || 8
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.queryBiases = []
        for (let i = 0; i < this.queries; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.queryBiases.push(
                this.addWeight(
                    `queryBiases${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
        }
        this.keyKernel = this.addWeight(
            'keyKernel',
            [units, this.projection],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.keyBias = this.addWeight(
            `keyBias`,
            [this.projection],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [units, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.valueBias = this.addWeight(
            `valueBias`,
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [units * this.queries, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.outputBias = this.addWeight(
            `outputBias`,
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const K = this.applyDense(inputs, this.keyKernel, this.keyBias)
            const V = this.applyDense(inputs, this.valueKernel, this.valueBias)

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.queries; i++) {
                const Q = this.applyDense(
                    inputs,
                    this.queryKernels[i],
                    this.queryBiases[i]
                )

                const scores = tf
                    .matMul(Q, K, false, true)
                    .div(tf.scalar(this.projection).sqrt())
                    .add(mask)

                let weights = scores.softmax()

                weights = kwargs['training']
                    ? tf.dropout(weights, this.dropout)
                    : weights

                const output = tf.matMul(weights, V)
                attentionOutputs.push(output)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            let outputs = this.applyDense(
                concatenatedOutputs,
                this.outputKernel,
                this.outputBias
            )

            outputs = this.residual.apply([inputs, outputs])

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getWeights() {
        return [
            ...this.queryKernels.map((kernel) => kernel.read()),
            ...this.queryBiases.map((kernel) => kernel.read()),
            this.keyKernel.read(),
            this.keyBias.read(),
            this.valueKernel.read(),
            this.valueBias.read(),
            this.outputKernel.read(),
            this.outputBias.read()
        ]
    }

    setWeights(weights) {
        for (let i = 0; i < this.queries; i++) {
            this.queryKernels[i].write(weights[i])
            this.queryBiases[i].write(weights[i])
        }
        this.keyKernel.write(weights[this.queries])
        this.keyBias.write(weights[this.queries + 1])
        this.valueKernel.write(weights[this.queries + 2])
        this.valueBias.write(weights[this.queries + 3])
        this.outputKernel.write(weights[this.queries + 4])
        this.outputBias.write(weights[this.queries + 5])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            projection: this.projection,
            queries: this.queries,
            dropout: this.dropout
        }
    }
}

class GroupedQueryAttention extends LayerBase {
    constructor(config) {
        super({ name: `gqa-${randomString()}`, ...config })
        this.projection = config.projection || 256
        this.heads = config.heads || 8
        this.queryRatio = config.queryRatio || 2
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.queryBiases = []
        this.keyKernels = []
        this.keyBiases = []
        this.valueKernels = []
        this.valueBiases = []

        for (let i = 0; i < this.heads; i++) {
            const queryKernels = []
            const queryBiases = []
            for (let j = 0; j < this.queryRatio; j++) {
                queryKernels.push(
                    this.addWeight(
                        `queryKernel-${i}-${j}`,
                        [units, this.projection],
                        'float32',
                        tf.initializers.glorotUniform(),
                        tf.regularizers.l2({ l2: 0.01 })
                    )
                )
                queryBiases.push(
                    this.addWeight(
                        `queryBias-${i}-${j}`,
                        [this.projection],
                        'float32',
                        tf.initializers.zeros(),
                        tf.regularizers.l2({ l2: 0.01 })
                    )
                )
            }
            this.queryKernels.push(queryKernels)
            this.queryBiases.push(queryBiases)

            this.keyKernels.push(
                this.addWeight(
                    `keyKernel-${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.keyBiases.push(
                this.addWeight(
                    `keyBiases-${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel-${i}`,
                    [units, units],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueBiases.push(
                this.addWeight(
                    `valueBiases-${i}`,
                    [units],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [units * this.heads * this.queryRatio, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputBias = this.addWeight(
            `outputBias`,
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.heads; i++) {
                const K = this.applyDense(
                    inputs,
                    this.keyKernels[i].read(),
                    this.keyBiases[i].read()
                )
                const V = this.applyDense(
                    inputs,
                    this.valueKernels[i].read(),
                    this.valueBiases[i].read()
                )

                for (let j = 0; j < this.queryRatio; j++) {
                    const Q = this.applyDense(
                        inputs,
                        this.queryKernels[i][j].read(),
                        this.queryBiases[i][j].read()
                    )

                    const scores = tf
                        .matMul(Q, K, false, true)
                        .div(tf.scalar(this.projection).sqrt())
                        .add(mask)

                    let weights = scores.softmax()

                    weights = kwargs['training']
                        ? tf.dropout(weights, this.dropout)
                        : weights

                    const output = tf.matMul(weights, V)
                    attentionOutputs.push(output)
                }
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            let outputs = this.applyDense(
                concatenatedOutputs,
                this.outputKernel.read(),
                this.outputBias.read()
            )

            outputs = this.rmsNorm(outputs)

            outputs = this.residual.apply([inputs, outputs])

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getWeights() {
        const weights = []

        for (let i = 0; i < this.heads; i++) {
            for (let j = 0; j < this.queryRatio; j++) {
                weights.push(this.queryKernels[i][j].read())
                weights.push(this.queryBiases[i][j].read())
            }
            weights.push(this.keyKernels[i].read())
            weights.push(this.keyBiases[i].read())
            weights.push(this.valueKernels[i].read())
            weights.push(this.valueBiases[i].read())
        }

        weights.push(this.outputKernel.read())
        weights.push(this.outputBias.read())

        return weights
    }

    setWeights(weights) {
        let index = 0

        for (let i = 0; i < this.heads; i++) {
            for (let j = 0; j < this.queryRatio; j++) {
                this.queryKernels[i][j].write(weights[index++])
                this.queryBiases[i][j].write(weights[index++])
            }
            this.keyKernels[i].write(weights[index++])
            this.keyBiases[i].write(weights[index++])
            this.valueKernels[i].write(weights[index++])
            this.valueBiases[i].write(weights[index++])
        }

        this.outputKernel.write(weights[index++])
        this.outputBias.write(weights[index])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            projection: this.projection,
            heads: this.heads,
            queryRatio: this.queryRatio,
            dropout: this.dropout
        }
    }
}

class NystromAttention extends LayerBase {
    constructor(config) {
        super({ name: `nystromformer-${randomString()}`, ...config })
        this.units = config.units || 64
        this.heads = config.heads || 8
        this.landmarks = config.landmarks || 64
    }

    build(inputShape) {
        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []

        const projectionSize = this.units / this.heads

        for (let i = 0; i < this.heads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [inputShape[inputShape.length - 1], projectionSize],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [inputShape[inputShape.length - 1], projectionSize],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [inputShape[inputShape.length - 1], projectionSize],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.units, this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const batchSize = inputs.shape[0]
            const sequenceLength = inputs.shape[1]

            const attentionOutputs = []

            for (let i = 0; i < this.heads; i++) {
                const Q = this.applyDense(inputs, this.queryKernels[i])
                const K = this.applyDense(inputs, this.keyKernels[i])
                const V = this.applyDense(inputs, this.valueKernels[i])

                const landmarks = this.sampleLandmarks(K, this.landmarks)

                const kernelMatrix = this.computeKernelMatrix(Q, landmarks)
                const attentionMatrix = this.computeAttentionMatrix(
                    kernelMatrix,
                    V,
                    landmarks
                )

                const output = attentionMatrix.reshape([
                    batchSize,
                    sequenceLength,
                    -1
                ])

                attentionOutputs.push(output)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            const outputs = this.applyDense(
                concatenatedOutputs,
                this.outputKernel
            )

            return this.residual.apply([inputs, outputs])
        })
    }

    sampleLandmarks(keys, numLandmarks) {
        const batchSize = keys.shape[0]
        const sequenceLength = keys.shape[1]

        const randomIndices = tf.randomUniform(
            [batchSize, numLandmarks],
            0,
            sequenceLength,
            'int32'
        )
        const landmarks = tf.gather(keys, randomIndices, 1)

        return landmarks
    }

    computeKernelMatrix(queries, landmarks) {
        const projectionSize = this.units / this.heads
        const batchSize = queries.shape[0]
        const sequenceLength = queries.shape[1]
        const numLandmarks = landmarks.shape[1]

        const flattenedQueries = queries.reshape([
            batchSize * sequenceLength,
            projectionSize
        ])
        const flattenedLandmarks = landmarks.reshape([
            batchSize * numLandmarks,
            projectionSize
        ])

        const queryProjections = tf.matMul(
            flattenedQueries,
            flattenedLandmarks,
            false,
            true
        )
        const kernelMatrix = tf.exp(
            queryProjections.div(tf.sqrt(tf.scalar(projectionSize)))
        )

        return kernelMatrix.reshape([
            batchSize,
            sequenceLength,
            batchSize,
            numLandmarks
        ])
    }

    computeAttentionMatrix(kernelMatrix, values, landmarks) {
        const valueLandmarks = tf.einsum('bnd,nl->bld', values, landmarks)
        const attentionMatrix = tf.einsum(
            'bnl,bld->bnd',
            kernelMatrix,
            valueLandmarks
        )

        return attentionMatrix
    }

    applyDense(x, kernel) {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
    }

    getWeights() {
        return [
            ...this.queryKernels.map((kernel) => kernel.read()),
            ...this.keyKernels.map((kernel) => kernel.read()),
            ...this.valueKernels.map((kernel) => kernel.read()),
            this.outputKernel.read()
        ]
    }

    setWeights(weights) {
        for (let i = 0; i < this.heads; i++) {
            this.queryKernels[i].write(weights[i])
            this.keyKernels[i].write(weights[this.heads + i])
            this.valueKernels[i].write(weights[this.heads * 2 + i])
        }
        this.outputKernel.write(weights[this.heads * 3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            heads: this.heads,
            landmarks: this.landmarks
        }
    }
}

class MultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super({ name: `mlp-${randomString()}`, ...config })
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.activation = config?.activation || 'relu'
        this.units = config.units || null
    }

    build(inputShape) {
        this.units = this.units ? this.units : inputShape[inputShape.length - 1]

        // Initialize dense layers for projection
        this.inProjKernel = this.addWeight(
            `inProjKernel`,
            [this.units, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.inProjBias = this.addWeight(
            `inProjBias`,
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        this.outProjKernel = this.addWeight(
            `outProjKernel`,
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outProjBias = this.addWeight(
            `outProjBias`,
            [this.units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        // Residual connections/skip connections are critical here
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand and contract projection via feedforward layers
            let outputs = this.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias.read()
            )

            outputs = this.rmsNorm(outputs)

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.applyDense(
                outputs,
                this.outProjKernel.read(),
                this.outProjBias.read()
            )

            outputs = this.residual.apply([inputs, outputs])

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getWeights() {
        return [
            this.inProjKernel.read(),
            this.inProjBias.read(),
            this.outProjKernel.read(),
            this.outProjBias.read()
        ]
    }

    setWeights(weights) {
        this.inProjKernel.write(weights[0])
        this.inProjBias.write(weights[1])
        this.outProjKernel.write(weights[2])
        this.outProjBias.write(weights[3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            activation: this.activation
        }
    }
}

class GatedLinearMLP extends MultiLayerPerceptron {
    constructor(config) {
        super({ name: `glu-${randomString()}`, ...config })
    }

    build(inputShape) {
        super.build(inputShape)

        this.gateProjKernel = this.addWeight(
            `gateProjKernel`,
            [this.units, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gateProjBias = this.addWeight(
            `gateProjBias`,
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand and contract projection via feedforward layers
            let proj = this.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias.read()
            )

            proj = this.rmsNorm(proj)

            proj = tf.layers
                .activation({ activation: this.activation })
                .apply(proj)

            let gate = this.applyDense(
                inputs,
                this.gateProjKernel.read(),
                this.gateProjBias.read()
            )

            gate = tf.layers.activation({ activation: 'sigmoid' }).apply(gate)

            const gatedOutput = tf.mul(proj, gate)

            let outputs = this.applyDense(
                gatedOutput,
                this.outProjKernel.read(),
                this.outProjBias.read()
            )

            outputs = this.residual.apply([inputs, outputs])

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getWeights() {
        return [
            ...super.getWeights(),
            this.gateProjKernel.read(),
            this.gateProjBias.read()
        ]
    }

    setWeights(weights) {
        super.setWeights(weights)
        this.gateProjKernel.write(weights[4])
        this.gateProjBias.write(weights[5])
    }
}

class MeltingMLP extends LayerBase {
    constructor(config) {
        super({ name: `mlp-${randomString()}`, ...config })
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.activation = config?.activation || 'relu'
        this.units = config.units || null
    }

    build(inputShape) {
        this.units = this.units ? this.units : inputShape[inputShape.length - 1]

        // Initialize dense layers for projection
        this.inProjKernel = this.addWeight(
            'inProjKernel',
            [this.units, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.inProjBias = this.addWeight(
            'inProjBias',
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        this.outProjKernel = this.addWeight(
            'outProjKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outProjBias = this.addWeight(
            'outProjBias',
            [this.units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        // Residual connections/skip connections are critical here
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Get the input dimensions
            const inputDim = inputs.shape[inputs.shape.length - 1]

            // Slice the weights based on the input dimensions
            const slicedInProjKernel = this.inProjKernel
                .read()
                .slice([0, 0], [inputDim, this.innerDim])
            const slicedOutProjKernel = this.outProjKernel
                .read()
                .slice([0, 0], [this.innerDim, inputDim])
            const slicedOutProjBias = this.outProjBias
                .read()
                .slice([0], [inputDim])

            // Expand and contract projection via feedforward layers
            let outputs = this.applyDense(
                inputs,
                slicedInProjKernel,
                this.inProjBias
            )

            outputs = this.rmsNorm(outputs)

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.applyDense(
                outputs,
                slicedOutProjKernel,
                slicedOutProjBias
            )

            outputs = this.residual.apply([inputs, outputs])

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getWeights() {
        return [
            this.inProjKernel.read(),
            this.inProjBias.read(),
            this.outProjKernel.read(),
            this.outProjBias.read()
        ]
    }

    setWeights(weights) {
        this.inProjKernel.write(weights[0])
        this.inProjBias.write(weights[1])
        this.outProjKernel.write(weights[2])
        this.outProjBias.write(weights[3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            activation: this.activation
        }
    }
}

class MixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.numExperts = config.numExperts
        this.hiddenDim = config.hiddenDim || 128
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[0][inputShape[0].length - 1]

        // Initialize gating network
        this.gatingHidden = this.addWeight(
            'gatingHidden',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingHiddenBias = this.addWeight(
            'gatingHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingKernel = this.addWeight(
            'gatingKernel',
            [this.hiddenDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingBias = this.addWeight(
            'gatingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const expertInputs = inputs.slice(1)
            inputs = inputs[0]

            // Gating network
            const gatingHidden = this.applyDense(
                inputs,
                this.gatingHidden,
                this.gatingHiddenBias
            )
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(gatingHidden)
            const expertWeights = this.applyDense(
                activatedGate,
                this.gatingKernel,
                this.gatingBias
            ).softmax()

            // Combine expert outputs using weighted sum
            const combinedOutput = expertInputs.reduce((prev, curr, i) => {
                const expertWeight = expertWeights.slice(
                    [0, 0, i],
                    [inputs.shape[0], inputs.shape[1], 1]
                )
                return prev.add(curr.mul(expertWeight))
            }, tf.zeros(expertInputs[0].shape))

            return combinedOutput
        })
    }

    computeOutputShape(inputShape) {
        return inputShape[0]
    }

    getWeights() {
        return [
            this.gatingHidden.read(),
            this.gatingHiddenBias.read(),
            this.gatingKernel.read(),
            this.gatingBias.read()
        ]
    }

    setWeights(weights) {
        this.gatingHidden.write(weights[0])
        this.gatingHiddenBias.write(weights[1])
        this.gatingKernel.write(weights[2])
        this.gatingBias.write(weights[3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            activation: this.activation
        }
    }
}

// class SparseMixtureOfExperts extends LayerBase {
//     constructor(config) {
//         super({ name: `moe-${randomString()}`, ...config })
//         this.numExperts = config.numExperts
//         this.hiddenDim = config.hiddenDim || 128
//         this.activation = config.activation || 'swish'
//     }

//     build(inputShape) {
//         const inputDim = inputShape[0][inputShape[0].length - 1]

//         // Initialize gating network
//         this.gatingHidden = this.addWeight(
//             'gatingHidden',
//             [inputDim, this.hiddenDim],
//             'float32',
//             tf.initializers.glorotNormal(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//         this.gatingHiddenBias = this.addWeight(
//             'gatingHiddenBias',
//             [this.hiddenDim],
//             'float32',
//             tf.initializers.zeros(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//         this.gatingKernel = this.addWeight(
//             'gatingKernel',
//             [this.hiddenDim, this.numExperts],
//             'float32',
//             tf.initializers.glorotNormal(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//         this.gatingBias = this.addWeight(
//             'gatingBias',
//             [this.numExperts],
//             'float32',
//             tf.initializers.zeros(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             const expertInputs = inputs.slice(1)
//             inputs = inputs[0]

//             // Gating network
//             const gatingHidden = this.applyDense(
//                 inputs,
//                 this.gatingHidden,
//                 this.gatingHiddenBias
//             )
//             const activatedGate = tf.layers
//                 .activation({ activation: this.activation })
//                 .apply(gatingHidden)
//             const expertWeights = this.applyDense(
//                 activatedGate,
//                 this.gatingKernel,
//                 this.gatingBias
//             ).softmax()

//             // Select 1 random expert for each time step
//             const selectedExpertIndices = this.selectRandomExpertsPerTimeStep(
//                 inputs.shape[1],
//                 expertInputs.length
//             )

//             // Slice and combine selected expert outputs
//             const selectedExpertOutputs = []
//             const selectedExpertWeights = []
//             for (let i = 0; i < inputs.shape[1]; i++) {
//                 const expertIndex = selectedExpertIndices[i]
//                 const expertOutput = expertInputs[expertIndex].slice(
//                     [0, i, 0],
//                     [inputs.shape[0], 1, expertInputs[0].shape[2]]
//                 )
//                 const expertWeight = expertWeights.slice(
//                     [0, i, expertIndex],
//                     [inputs.shape[0], 1, 1]
//                 )
//                 selectedExpertOutputs.push(expertOutput)
//                 selectedExpertWeights.push(expertWeight)
//             }

//             // Concatenate the selected expert outputs and weights along the time dimension
//             const combinedOutput = tf.concat(selectedExpertOutputs, 1)
//             const combinedWeights = tf.concat(selectedExpertWeights, 1)

//             // Multiply the combined output with the combined weights
//             const weightedOutput = combinedOutput.mul(combinedWeights)

//             return weightedOutput
//         })
//     }

//     selectRandomExpertsPerTimeStep(timeSteps, numExperts) {
//         const selectedExperts = []
//         for (let i = 0; i < timeSteps; i++) {
//             const randomExpertIndex = Math.floor(Math.random() * numExperts)
//             selectedExperts.push(randomExpertIndex)
//         }
//         return selectedExperts
//     }

//     computeOutputShape(inputShape) {
//         return inputShape[0]
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             numExperts: this.numExperts,
//             hiddenDim: this.hiddenDim,
//             activation: this.activation
//         }
//     }
// }

class SparseMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.numExperts = config.numExperts
        this.topK = config.topK || 2
        this.hiddenDim = config.hiddenDim || 128
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[0][inputShape[0].length - 1]

        // Initialize gating network
        this.gatingHidden = this.addWeight(
            'gatingHidden',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingHiddenBias = this.addWeight(
            'gatingHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingKernel = this.addWeight(
            'gatingKernel',
            [this.hiddenDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingBias = this.addWeight(
            'gatingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const expertInputs = inputs.slice(1)
            inputs = inputs[0]

            // Gating network
            const gatingHidden = this.applyDense(
                inputs,
                this.gatingHidden.read(),
                this.gatingHiddenBias.read()
            )
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(gatingHidden)
            const expertWeights = this.applyDense(
                activatedGate,
                this.gatingKernel.read(),
                this.gatingBias.read()
            ).softmax()

            // Randomly select a subset of experts
            const selectedExpertIndices = this.selectRandomExperts(expertInputs)

            // Slice the expert weights based on the selected expert indices
            const selectedExpertWeights = this.sliceExpertWeights(
                expertWeights,
                selectedExpertIndices
            )

            // Slice and combine selected expert outputs
            const selectedExpertOutputs = []
            selectedExpertIndices.map((expertIndex) => {
                selectedExpertOutputs.push(expertInputs[expertIndex])
            })

            // Combine expert outputs using weighted sum
            const combinedOutput = selectedExpertOutputs.reduce(
                (prev, curr, i) => {
                    const expertWeight = selectedExpertWeights.slice(
                        [0, 0, i],
                        [inputs.shape[0], inputs.shape[1], 1]
                    )
                    return prev.add(curr.mul(expertWeight))
                },
                tf.zeros(expertInputs[0].shape)
            )

            return combinedOutput
        })
    }

    selectRandomExperts(expertInputs) {
        const numExperts = expertInputs.length
        const expertIndices = tf.util.createShuffledIndices(numExperts)
        return expertIndices.slice(0, this.topK)
    }

    sliceExpertWeights(expertWeights, selectedExpertIndices) {
        const selectedWeights = []
        selectedExpertIndices.forEach((expertIndex) => {
            const expertSlice = expertWeights.slice(
                [0, 0, expertIndex],
                [expertWeights.shape[0], expertWeights.shape[1], 1]
            )
            selectedWeights.push(expertSlice)
        })
        return tf.concat(selectedWeights, -1)
    }

    computeOutputShape(inputShape) {
        return inputShape[0]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            activation: this.activation,
            topK: this.topK
        }
    }
}

// class SparseMixtureOfExperts extends LayerBase {
//     constructor(config) {
//         super({ name: `moe-${randomString()}`, ...config })
//         this.numExperts = config.numExperts
//         this.topK = config.topK || 2
//         this.hiddenDim = config.hiddenDim || 128
//         this.activation = config.activation || 'swish'
//     }

//     build(inputShape) {
//         const inputDim = inputShape[0][inputShape[0].length - 1]

//         // Initialize gating network
//         this.gatingHidden = this.addWeight(
//             'gatingHidden',
//             [inputDim, this.hiddenDim],
//             'float32',
//             tf.initializers.glorotNormal(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//         this.gatingHiddenBias = this.addWeight(
//             'gatingHiddenBias',
//             [this.hiddenDim],
//             'float32',
//             tf.initializers.zeros(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//         this.gatingKernel = this.addWeight(
//             'gatingKernel',
//             [this.hiddenDim, this.numExperts],
//             'float32',
//             tf.initializers.glorotNormal(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//         this.gatingBias = this.addWeight(
//             'gatingBias',
//             [this.numExperts],
//             'float32',
//             tf.initializers.zeros(),
// tf.regularizers.l2({ l2: 0.01 })
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             const expertInputs = inputs.slice(1)
//             inputs = inputs[0]

//             const timeSteps = inputs.shape[1]
//             const batchSize = inputs.shape[0]
//             const outputDim = expertInputs[0].shape[2]

//             // Gating network
//             const gatingHidden = this.applyDense(
//                 inputs,
//                 this.gatingHidden,
//                 this.gatingHiddenBias
//             )
//             const activatedGate = tf.layers
//                 .activation({ activation: this.activation })
//                 .apply(gatingHidden)
//             const expertWeights = this.applyDense(
//                 activatedGate,
//                 this.gatingKernel,
//                 this.gatingBias
//             ).softmax()

//             // Randomly select a subset of experts
//             const selectedExpertIndices = this.selectRandomExperts(expertInputs)

//             // Create separate arrays for each expert
//             const expertOutputArrays = Array.from(
//                 { length: this.topK },
//                 () => []
//             )
//             const expertWeightArrays = Array.from(
//                 { length: this.topK },
//                 () => []
//             )

//             for (let t = 0; t < timeSteps; t++) {
//                 // Slice the expert weights based on the selected expert indices
//                 const slicedExpertWeights = this.sliceExpertWeights(
//                     expertWeights.slice(
//                         [0, t, 0],
//                         [batchSize, 1, this.numExperts]
//                     ),
//                     selectedExpertIndices
//                 )

//                 // Slice the selected expert outputs and push them to separate arrays
//                 for (let i = 0; i < this.topK; i++) {
//                     const expertIndex = selectedExpertIndices[i]
//                     const expertOutput = expertInputs[expertIndex].slice(
//                         [0, t, 0],
//                         [batchSize, 1, outputDim]
//                     )
//                     expertOutputArrays[i].push(expertOutput)
//                     expertWeightArrays[i].push(
//                         slicedExpertWeights.slice([0, 0, i], [batchSize, 1, 1])
//                     )
//                 }
//             }

//             // Concatenate the expert outputs and weights along the time dimension
//             const concatenatedExpertOutputs = expertOutputArrays.map(
//                 (outputs) => tf.concat(outputs, 1)
//             )
//             const concatenatedExpertWeights = expertWeightArrays.map(
//                 (weights) => tf.concat(weights, 1)
//             )

//             // Compute the weighted sum of expert outputs
//             const weightedExpertOutputs = concatenatedExpertOutputs.map(
//                 (outputs, i) => outputs.mul(concatenatedExpertWeights[i])
//             )
//             const combinedOutput = weightedExpertOutputs.reduce(
//                 (prev, curr) => prev.add(curr),
//                 tf.zeros([batchSize, timeSteps, outputDim])
//             )

//             return combinedOutput
//         })
//     }

//     selectRandomExperts(expertInputs) {
//         const numExperts = expertInputs.length
//         const expertIndices = tf.util.createShuffledIndices(numExperts)
//         return expertIndices.slice(0, this.topK)
//     }

//     sliceExpertWeights(expertWeights, selectedExpertIndices) {
//         const selectedWeights = []
//         selectedExpertIndices.forEach((expertIndex) => {
//             const expertSlice = expertWeights.slice(
//                 [0, 0, expertIndex],
//                 [expertWeights.shape[0], expertWeights.shape[1], 1]
//             )
//             selectedWeights.push(expertSlice)
//         })
//         return tf.concat(selectedWeights, -1)
//     }

//     computeOutputShape(inputShape) {
//         return inputShape[0]
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             numExperts: this.numExperts,
//             hiddenDim: this.hiddenDim,
//             activation: this.activation,
//             topK: this.topK
//         }
//     }
// }

class TransientMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.numExperts = config.numExperts
        this.topK = config.topK || 2
        this.hiddenDim = config.hiddenDim || 128
        this.activation = config.activation || 'swish'
        this.expertArgs = config.expertArgs || {
            type: 'SelfAttention',
            projection: 64
        }
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.experts = this.createExperts()
        this.experts.map((expert) => {
            expert.build(inputShape)
            // this._trainableWeights.push(...expert.trainableWeights)
        })

        // Initialize gating network
        this.gatingHidden = this.addWeight(
            'gatingHidden',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingHiddenBias = this.addWeight(
            'gatingHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingKernel = this.addWeight(
            'gatingKernel',
            [this.hiddenDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingBias = this.addWeight(
            'gatingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Gating network
            const gatingHidden = this.applyDense(
                inputs,
                this.gatingHidden.read(),
                this.gatingHiddenBias.read()
            )
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(gatingHidden)
            const expertWeights = this.applyDense(
                activatedGate,
                this.gatingKernel.read(),
                this.gatingBias.read()
            ).softmax()

            // Randomly select a subset of experts
            const selectedExpertIndices = this.selectRandomExperts(this.experts)

            // // Slice the expert weights based on the selected expert indices
            const selectedExpertWeights = this.sliceExpertWeights(
                expertWeights,
                selectedExpertIndices
            )

            // Perform inference for each expert
            const expertOutputs = []
            selectedExpertIndices.map((index) =>
                expertOutputs.push(this.experts[index].apply(inputs))
            )

            // Combine expert outputs using weighted sum
            const combinedOutput = expertOutputs.reduce((prev, curr, i) => {
                const expertWeight = selectedExpertWeights.slice(
                    [0, 0, i],
                    [inputs.shape[0], inputs.shape[1], 1]
                )
                return prev.add(curr.mul(expertWeight))
            }, tf.zeros(inputs.shape))

            return combinedOutput
        })
    }

    selectRandomExperts(experts) {
        const numExperts = experts.length
        const expertIndices = tf.util.createShuffledIndices(numExperts)
        return expertIndices.slice(0, this.topK)
    }

    queryExperts(expertWeights, selectedExpertIndices) {
        const selectedWeights = []
        selectedExpertIndices.forEach((expertIndex) => {
            const expertSlice = expertWeights.slice(
                [0, 0, expertIndex],
                [expertWeights.shape[0], expertWeights.shape[1], 1]
            )
            selectedWeights.push(expertSlice)
        })
        return tf.concat(selectedWeights, -1)
    }

    sliceExpertWeights(expertWeights, selectedExpertIndices) {
        const selectedWeights = []
        selectedExpertIndices.forEach((expertIndex) => {
            const expertSlice = expertWeights.slice(
                [0, 0, expertIndex],
                [expertWeights.shape[0], expertWeights.shape[1], 1]
            )
            selectedWeights.push(expertSlice)
        })
        return tf.concat(selectedWeights, -1)
    }

    createExperts() {
        const experts = []
        for (let i = 0; i < this.numExperts; i++) {
            experts.push(this.findLayer(this.expertArgs.type)(this.expertArgs))
        }
        return experts
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getWeights() {
        const expertWeights = this.experts.map((expert) => expert.getWeights())
        return [
            this.gatingHidden.read(),
            this.gatingHiddenBias.read(),
            this.gatingKernel.read(),
            this.gatingBias.read(),
            ...expertWeights.flat()
        ]
    }

    setWeights(weights) {
        this.gatingHidden.write(weights[0])
        this.gatingHiddenBias.write(weights[1])
        this.gatingKernel.write(weights[2])
        this.gatingBias.write(weights[3])

        const expertWeights = weights.slice(4)
        this.experts.forEach((expert, i) => {
            expert.setWeights(
                expertWeights.slice(
                    i * expert.countParams(),
                    (i + 1) * expert.countParams()
                )
            )
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            activation: this.activation,
            topK: this.topK,
            expertArgs: this.expertArgs
        }
    }
}

class AdaptiveMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.numExperts = config.numExperts
        this.topK = config.topK || 2
        this.switchingDim = config?.switchingDim || 64
        this.activation = config.activation || 'swish'
        this.expertArgs = config.expertArgs || {
            type: 'SelfAttention',
            projection: 64
        }
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        // this.experts = this.createExperts()

        this.experts = []
        for (let i = 0; i < this.numExperts; i++) {
            const expert = new Expert({ ...this.expertArgs, inputShape })
            this.experts.push(expert.model)
        }

        // Initialize switching network
        this.switchingHidden = this.addWeight(
            'switchingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingHiddenBias = this.addWeight(
            'switchingHiddenBias',
            [this.switchingDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingKernel = this.addWeight(
            'switchingKernel',
            [this.switchingDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingBias = this.addWeight(
            'switchingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputProjection = this.addWeight(
            'outputProjection',
            [
                this.topK * this.experts[0].computeOutputShape(inputShape)[2],
                inputDim
            ],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Switching network
            const switchingHidden = this.applyDense(
                inputs,
                this.switchingHidden.read(),
                this.switchingHiddenBias.read()
            )
            const switchingActivated = tf.layers
                .activation({ activation: this.activation })
                .apply(switchingHidden)
            const switchingScores = this.applyDense(
                switchingActivated,
                this.switchingKernel.read(),
                this.switchingBias.read()
            )

            // Select top-k experts for each batch
            const [batchSize, timeSteps, numExperts] = switchingScores.shape
            const linearWeights = tf
                .linspace(1, 2, timeSteps)
                .expandDims(0)
                .expandDims(-1)
            const weightedAvgScores = switchingScores
                .mul(linearWeights)
                .sum(1)
                .div(linearWeights.sum())

            const selectedExpertIndices =
                this.selectTopExperts(weightedAvgScores)

            // Predict on top-k experts, for every batch
            const batchOutputs = []
            for (let i = 0; i < inputs.shape[0]; i++) {
                const batchInputs = inputs.slice([i, 0], [1, -1])
                const expertOutputs = []
                for (let j = 0; j < this.topK; j++) {
                    const expertIndex = selectedExpertIndices[i][j]
                    const expertOutput =
                        this.experts[expertIndex].apply(batchInputs)
                    expertOutputs.push(expertOutput)
                }
                const concatenatedOutput = tf.concat(expertOutputs, -1)
                batchOutputs.push(concatenatedOutput)
            }

            // Concat expert outputs, and project them into the proper dimension
            const outputProjected = this.applyDense(
                tf.concat(batchOutputs, 0),
                this.outputProjection.read()
            )

            return outputProjected
        })
    }

    // createExperts() {
    //     const experts = []
    //     for (let i = 0; i < this.numExperts; i++) {
    //         experts.push(this.findLayer(this.expertArgs.type)(this.expertArgs))
    //     }
    //     return experts
    // }

    selectTopExperts(switchingScores) {
        const topKIndices = tf.topk(switchingScores, this.topK).indices
        return topKIndices.arraySync()
    }

    sliceExpertWeights(expertWeights, selectedExpertIndices) {
        return tf.tidy(() => {
            const batchSize = expertWeights.shape[0]
            const sequenceLength = expertWeights.shape[1]

            const selectedWeights = []
            for (let i = 0; i < batchSize; i++) {
                const batchWeights = expertWeights.slice([i, 0, 0], [1, -1, -1])
                const batchSelectedWeights = []
                for (let j = 0; j < this.topK; j++) {
                    const expertIndex = selectedExpertIndices[i][j]
                    batchSelectedWeights.push(
                        batchWeights.slice(
                            [0, 0, expertIndex],
                            [1, sequenceLength, 1]
                        )
                    )
                }
                selectedWeights.push(tf.concat(batchSelectedWeights, 2))
            }
            return tf.concat(selectedWeights, 0)
        })
    }

    getWeights() {
        return [
            this.switchingHidden.read(),
            this.switchingHiddenBias.read(),
            this.switchingKernel.read(),
            this.switchingBias.read(),
            this.outputProjection.read()
        ]
    }

    setWeights(weights) {
        this.switchingHidden.write(weights[0])
        this.switchingHiddenBias.write(weights[1])
        this.switchingKernel.write(weights[2])
        this.switchingBias.write(weights[3])
        this.outputProjection.write(weights[4])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            switchingDim: this.switchingDim,
            activation: this.activation,
            topK: this.topK,
            expertArgs: this.expertArgs
        }
    }
}

class KANLayer extends LayerBase {
    constructor(config) {
        super({ name: `kan-${randomString()}`, ...config })
        this.inDim = config.inDim
        this.outDim = config.outDim
        this.size = this.outDim * this.inDim
        this.num = config.num || 5
        this.k = config.k || 3
        this.noiseScale = config.noiseScale || 0.1
        this.scaleBase = config.scaleBase || 1.0
        this.scaleSp = config.scaleSp || 1.0
        this.baseFun = config.baseFun || ((x) => tf.relu(x))
        this.gridEps = config.gridEps || 0.02
        this.gridRange = config.gridRange || [-1, 1]
        this.spTrainable =
            config.spTrainable !== undefined ? config.spTrainable : true
        this.sbTrainable =
            config.sbTrainable !== undefined ? config.sbTrainable : true
    }

    build(inputShape) {
        // Initialize grid
        this.grid = this.addWeight(
            'grid',
            [this.size, this.num + 1],
            'float32',
            tf.initializers.glorotUniform()
        )

        // Initialize coefficients
        this.coef = this.addWeight(
            'coef',
            [this.size, this.num + this.k],
            'float32',
            tf.initializers.glorotUniform()
        )

        // Initialize scales
        this.scaleBase = this.addWeight(
            'scaleBase',
            [this.size],
            'float32',
            tf.initializers.constant({ value: this.scaleBase }),
            undefined,
            this.sbTrainable
        )

        this.scaleSp = this.addWeight(
            'scaleSp',
            [this.size],
            'float32',
            tf.initializers.constant({ value: this.scaleSp }),
            undefined,
            this.spTrainable
        )

        // Initialize mask
        this.mask = this.addWeight(
            'mask',
            [this.size],
            'float32',
            tf.initializers.ones(),
            undefined,
            false
        )

        this.weightSharing = tf.range(0, this.size).cast('int32')
        this.lockCounter = 0
        this.lockId = tf.zeros([this.size])
    }

    B_batch(x, grid, k = 0, extend = true) {
        return tf.tidy(() => {
            console.log('B_batch input x shape:', x.shape)
            console.log('B_batch input grid shape:', grid.shape)

            const extendGrid = (grid, kExtend = 0) => {
                console.log('extendGrid input shape:', grid.shape)

                const lastCol = grid.slice([0, grid.shape[1] - 1], [-1, 1])
                const firstCol = grid.slice([0, 0], [-1, 1])
                console.log('lastCol shape:', lastCol.shape)
                console.log('firstCol shape:', firstCol.shape)

                const h = lastCol.sub(firstCol).div(grid.shape[1] - 1)
                console.log('h shape:', h.shape)

                let extendedGrid = grid
                for (let i = 0; i < kExtend; i++) {
                    const leftExtension = firstCol.sub(h)
                    const rightExtension = lastCol.add(h)
                    console.log('leftExtension shape:', leftExtension.shape)
                    console.log('rightExtension shape:', rightExtension.shape)

                    extendedGrid = tf.concat(
                        [leftExtension, extendedGrid, rightExtension],
                        1
                    )
                    console.log(
                        'extendedGrid shape after extension:',
                        extendedGrid.shape
                    )
                }
                return extendedGrid
            }

            if (extend) {
                grid = extendGrid(grid, k)
            }

            console.log('Extended grid shape:', grid.shape)

            // Reshape x and grid to 2D, maintaining the last dimension
            const x2d = x.reshape([-1, x.shape[x.shape.length - 1]])
            const grid2d = grid.reshape([-1, grid.shape[grid.shape.length - 1]])

            console.log('Reshaped x2d shape:', x2d.shape)
            console.log('Reshaped grid2d shape:', grid2d.shape)

            const sliceGrid = (g, start, size) => {
                size = Math.max(0, Math.min(size, g.shape[0] - start))
                return g.slice([start, 0], [size, g.shape[1]])
            }

            // Initialize B with the base case (k = 0)
            let gridSlice1 = sliceGrid(grid2d, 0, grid2d.shape[0] - 1)
            let gridSlice2 = sliceGrid(grid2d, 1, grid2d.shape[0] - 1)

            // Ensure gridSlice1 and gridSlice2 have the same number of rows as x2d
            const paddings = [
                [0, x2d.shape[0] - gridSlice1.shape[0]],
                [0, 0]
            ]
            gridSlice1 = tf.pad(gridSlice1, paddings)
            gridSlice2 = tf.pad(gridSlice2, paddings)

            console.log('Adjusted grid slice 1 shape:', gridSlice1.shape)
            console.log('Adjusted grid slice 2 shape:', gridSlice2.shape)
            console.log('x2d shape:', x2d.shape)

            const compareGreaterEqual = (x, y) =>
                tf
                    .gather(x, tf.range(0, x.shape[1], undefined, 'int32'), 1)
                    .greaterEqual(y)
            const compareLess = (x, y) =>
                tf
                    .gather(x, tf.range(0, x.shape[1], undefined, 'int32'), 1)
                    .less(y)

            let B = compareGreaterEqual(x2d, gridSlice1)
                .logicalAnd(compareLess(x2d, gridSlice2))
                .toFloat()

            console.log('Initial B shape:', B.shape)

            // Iteratively compute B for increasing k
            for (let i = 1; i <= k; i++) {
                let slice1 = sliceGrid(grid2d, 0, grid2d.shape[0] - (i + 1))
                let slice2 = sliceGrid(grid2d, i, grid2d.shape[0] - i)
                let slice3 = sliceGrid(grid2d, i + 1, grid2d.shape[0] - (i + 1))
                let slice4 = sliceGrid(grid2d, 1, grid2d.shape[0] - i)

                // Ensure slice1, slice2, slice3, and slice4 have the same number of rows as x2d
                slice1 = tf.pad(slice1, paddings)
                slice2 = tf.pad(slice2, paddings)
                slice3 = tf.pad(slice3, paddings)
                slice4 = tf.pad(slice4, paddings)

                console.log(
                    `Iteration ${i} - Slice shapes:`,
                    slice1.shape,
                    slice2.shape,
                    slice3.shape,
                    slice4.shape
                )

                const B_slice = B.slice([0, 0], [B.shape[0] - 1, B.shape[1]])
                console.log(`Iteration ${i} - B_slice shape:`, B_slice.shape)

                const sub1 = tf
                    .gather(
                        x2d,
                        tf.range(0, x2d.shape[1], undefined, 'int32'),
                        1
                    )
                    .sub(slice1)
                console.log(`Iteration ${i} - sub1 shape:`, sub1.shape)

                const sub2 = slice2.sub(slice1)
                console.log(`Iteration ${i} - sub2 shape:`, sub2.shape)

                const div1 = sub1.div(sub2)
                console.log(`Iteration ${i} - div1 shape:`, div1.shape)

                const term1 = div1.mul(B_slice)
                console.log(`Iteration ${i} - term1 shape:`, term1.shape)

                const sub3 = slice3.sub(
                    tf.gather(
                        x2d,
                        tf.range(0, x2d.shape[1], undefined, 'int32'),
                        1
                    )
                )
                console.log(`Iteration ${i} - sub3 shape:`, sub3.shape)

                const sub4 = slice3.sub(slice4)
                console.log(`Iteration ${i} - sub4 shape:`, sub4.shape)

                const div2 = sub3.div(sub4)
                console.log(`Iteration ${i} - div2 shape:`, div2.shape)

                const term2 = div2.mul(
                    B.slice([1, 0], [B.shape[0] - 1, B.shape[1]])
                )
                console.log(`Iteration ${i} - term2 shape:`, term2.shape)

                B = term1.add(term2)
                console.log(`Iteration ${i} - B shape:`, B.shape)
            }

            // Reshape B back to match x's original shape
            const B3d = B.reshape(x.shape)
            console.log('B_batch output shape:', B3d.shape)
            return B3d
        })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batch, timeSteps, features] = inputs.shape

            console.log('Input shape:', [batch, timeSteps, features])

            // Reshape inputs to 2D: [batch * timeSteps, features]
            const flatInputs = inputs.reshape([-1, features])

            console.log('Flat inputs shape:', flatInputs.shape)

            // Apply KAN operations on flattened input
            const x = tf
                .einsum('ij,k->ikj', flatInputs, tf.ones([this.outDim]))
                .reshape([batch * timeSteps, this.size])
                .transpose()

            console.log('X shape after einsum:', x.shape)

            const preacts = x
                .transpose()
                .reshape([batch * timeSteps, this.outDim, this.inDim])

            console.log('Preacts shape:', preacts.shape)

            const base = this.baseFun(x).transpose()

            console.log('Base shape:', base.shape)

            const y = this.coef2curve(
                x,
                this.grid.read().gather(this.weightSharing),
                this.coef.read().gather(this.weightSharing),
                this.k
            ).transpose()

            console.log('Y shape:', y.shape)

            const postspline = y.reshape([
                batch * timeSteps,
                this.outDim,
                this.inDim
            ])

            console.log('Postspline shape:', postspline.shape)

            const output = this.scaleBase
                .read()
                .expandDims(0)
                .mul(base)
                .add(this.scaleSp.read().expandDims(0).mul(y))

            console.log('Output shape:', output.shape)

            const maskedOutput = this.mask.read().expandDims(0).mul(output)

            console.log('Masked output shape:', maskedOutput.shape)

            const postacts = maskedOutput.reshape([
                batch * timeSteps,
                this.outDim,
                this.inDim
            ])

            console.log('Postacts shape:', postacts.shape)

            const result = maskedOutput
                .reshape([batch * timeSteps, this.outDim, this.inDim])
                .sum(2)

            console.log('Result shape:', result.shape)

            // Reshape result back to 3D: [batch, timeSteps, outDim]
            const reshapedResult = result.reshape([
                batch,
                timeSteps,
                this.outDim
            ])

            console.log('Reshaped result shape:', reshapedResult.shape)

            return [
                reshapedResult,
                preacts.reshape([batch, timeSteps, this.outDim, this.inDim]),
                postacts.reshape([batch, timeSteps, this.outDim, this.inDim]),
                postspline.reshape([batch, timeSteps, this.outDim, this.inDim])
            ]
        })
    }

    coef2curve(xEval, grid, coef, k) {
        return tf.tidy(() => {
            return tf.einsum('ij,ijk->ik', coef, this.B_batch(xEval, grid, k))
        })
    }

    curve2coef(xEval, yEval, grid, k) {
        return tf.tidy(() => {
            const mat = this.B_batch(xEval, grid, k).transpose([0, 2, 1])
            return tf.linalg.lstSquares(mat, yEval.expandDims(2)).squeeze(2)
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            inDim: this.inDim,
            outDim: this.outDim,
            num: this.num,
            k: this.k,
            noiseScale: this.noiseScale,
            scaleBase: this.scaleBase,
            scaleSp: this.scaleSp,
            baseFun: this.baseFun,
            gridEps: this.gridEps,
            gridRange: this.gridRange,
            spTrainable: this.spTrainable,
            sbTrainable: this.sbTrainable
        }
    }
}

class KolmogorovArnoldNetwork extends LayerBase {
    constructor(config) {
        super({ name: `kan-${randomString()}`, ...config })
        this.degree = config?.degree || 3
        this.units = config?.units || 256
        this.gridPoints = tf.linspace(0.0, 1.0, this.units + 1).arraySync()
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || false
        this.supportsMasking = true
    }

    build(inputShape) {
        this.coefficients = this.addWeight(
            'coefficients',
            [this.gridPoints.length + this.degree - 1],
            'float32',
            tf.initializers.randomNormal({ mean: 0, stddev: 1 })
        )

        this.bSplines = this._createBSplines()

        if (this.epsilon) {
            this.layernorm = tf.layers.layerNormalization({
                epsilon: this.epsilon
            })
        }

        this.residual = customLayers.ResidualConnection()
    }

    _createBSplines() {
        const knots = [
            ...Array(this.degree).fill(this.gridPoints[0]),
            ...this.gridPoints,
            ...Array(this.degree).fill(
                this.gridPoints[this.gridPoints.length - 1]
            )
        ]

        const bSplines = []
        for (let i = 0; i < this.gridPoints.length + this.degree - 1; i++) {
            bSplines.push(
                this._bSplineBasis(knots.slice(i, i + this.degree + 1))
            )
        }

        return bSplines
    }

    _bSplineBasis(knots) {
        return (x) => {
            let y = tf.zeros([x.shape[0]])
            for (let i = 0; i < knots.length - this.degree; i++) {
                y = tf.add(
                    y,
                    this._coxDeBoor(
                        knots.slice(i, i + this.degree + 1),
                        x,
                        this.degree
                    )
                )
            }
            return y
        }
    }

    _coxDeBoor(knots, x, degree) {
        if (degree === 0) {
            if (knots.length === 1) {
                const diff = tf.abs(tf.sub(x, knots[0]))
                const eps = 1e-8
                const inRange = tf.sigmoid(tf.mul(tf.sub(eps, diff), 1e8))
                return tf.mul(inRange, tf.ones([x.shape[0]]))
            } else {
                const lowerBound = tf.sigmoid(tf.mul(tf.sub(x, knots[0]), 1e8))
                const upperBound = tf.sigmoid(tf.mul(tf.sub(knots[1], x), 1e8))
                const inRange = tf.mul(lowerBound, upperBound)
                return tf.mul(inRange, tf.ones([x.shape[0]]))
            }
        }

        if (knots.length === 1) {
            return tf.zerosLike(x)
        }

        const denom1 = tf.sub(knots[degree], knots[0])
        const eps = 1e-8
        const isDenom1Positive = tf.sigmoid(tf.mul(denom1, 1e8))
        const term1 = tf.mul(
            tf.mul(
                tf.sub(x, knots[0]),
                this._coxDeBoor(knots.slice(0, -1), x, degree - 1)
            ),
            tf.div(isDenom1Positive, tf.add(denom1, eps))
        )

        let term2
        if (degree + 1 < knots.length) {
            const denom2 = tf.sub(knots[degree + 1], knots[1])
            const isDenom2Positive = tf.sigmoid(tf.mul(denom2, 1e8))
            term2 = tf.mul(
                tf.mul(
                    tf.sub(knots[degree + 1], x),
                    this._coxDeBoor(knots.slice(1), x, degree - 1)
                ),
                tf.div(isDenom2Positive, tf.add(denom2, eps))
            )
        } else {
            term2 = tf.zerosLike(x)
        }

        return tf.add(term1, term2)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const mulResults = this.bSplines.map((bSpline, i) => {
                const coeffSlice = this.coefficients.read().slice([i], [1])

                const bSplineResult = bSpline(inputs)

                return tf.mul(coeffSlice, bSplineResult)
            })

            const phiX = tf.addN(mulResults)

            let outputs = phiX

            if (kwargs['training']) {
                outputs = tf.dropout(outputs, this.dropout)
            }

            if (this.layernorm) {
                outputs = this.layernorm.apply(outputs)
            }

            return this.residual.apply([inputs, outputs])
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getWeights() {
        return [this.coefficients.read()]
    }

    setWeights(weights) {
        this.coefficients.write(weights[0])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            gridPoints: this.gridPoints,
            degree: this.degree,
            dropout: this.dropout
        }
    }
}

class DenseMultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super({ name: `mlp-${randomString()}`, ...config })
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || false
        this.activation = config?.activation || 'relu'
        this.supportsMasking = true
    }

    build(inputShape) {
        // Initialize dense layers for projection
        this.inProj = tf.layers.dense({
            units: this.innerDim,
            inputShape: inputShape,
            activation: 'linear'
        })
        this.denseHTo4H = tf.layers.dense({
            units: this.innerDim * 4,
            inputShape: [this.innerDim],
            activation: this.activation
        })
        this.dense4HToH = tf.layers.dense({
            units: this.innerDim,
            inputShape: [this.innerDim * 4],
            activation: 'linear'
        })
        this.outProj = tf.layers.dense({
            units: this.units,
            inputShape: [this.innerDim],
            activation: 'linear'
        })

        // Initialize layer normalization
        if (this.epsilon) {
            this.layernorm = tf.layers.layerNormalization({
                epsilon: this.epsilon
            })
        }

        // Residual connections/skip connections are critical here
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand and contract projection via feedforward layers
            let outputs = this.inProj.apply(inputs)
            outputs = this.denseHTo4H.apply(outputs)
            outputs = this.dense4HToH.apply(outputs)
            outputs = this.outProj.apply(outputs)
            // If training, apply residual dropout
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs
            // Apply layer norm
            if (this.layernorm) outputs = this.layernorm.apply(outputs)
            // Apply skip connection
            return this.residual.apply([inputs, outputs])
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout
        }
    }
}

class Autoencoder extends LayerBase {
    constructor(config) {
        super({ name: `dia-${randomString()}`, ...config })
        this.innerDim = config?.innerDim || 1024
        this.bottleneck = config?.bottleneck || 128
        this.outputDim = config?.outputDim || 256
        this.encoderActivation = config?.encoderActivation || 'relu'
        this.decoderActivation = config?.decoderActivation || 'relu'
        this.variational = config?.variational || false
        this.beta = config?.beta || 1.0
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        const multiplier = this.variational ? 2 : 1

        // Initialize dense layers for encoder
        this.encoderKernel1 = this.addWeight(
            'encoderKernel1',
            [inputDim, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.encoderBias1 = this.addWeight(
            'encoderBias1',
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.encoderKernel2 = this.addWeight(
            'encoderKernel2',
            [this.innerDim, this.bottleneck * multiplier],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.encoderBias2 = this.addWeight(
            'encoderBias2',
            [this.bottleneck * multiplier],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        // Initialize dense layers for decoder
        this.decoderKernel1 = this.addWeight(
            'decoderKernel1',
            [this.bottleneck, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.decoderBias1 = this.addWeight(
            'decoderBias1',
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.decoderKernel2 = this.addWeight(
            'decoderKernel2',
            [this.innerDim, this.outputDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.decoderBias2 = this.addWeight(
            'decoderBias2',
            [this.outputDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    computeVariance(inputs, kwargs) {
        // Compute the mean and log-variance
        const [mean, logVar] = tf.split(inputs, 2, -1)
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
            this.extraLoss = tf.keep(klDivergence)
        }

        // Sample from the latent space using the reparameterization trick
        const epsilon = tf.randomNormal(mean.shape)
        return mean.add(epsilon.mul(expLogVar.sqrt()))
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Encode the inputs to the bottleneck representation
            let outputs = this.applyDense(
                inputs,
                this.encoderKernel1.read(),
                this.encoderBias1.read()
            )

            outputs = tf.layers
                .activation({ activation: this.encoderActivation })
                .apply(outputs)

            outputs = this.applyDense(
                outputs,
                this.encoderKernel2.read(),
                this.encoderBias2.read()
            )

            if (this.variational) {
                outputs = this.computeVariance(outputs, kwargs)
            }

            // Decode the bottleneck representation to the output dimensionality
            outputs = this.applyDense(
                outputs,
                this.decoderKernel1.read(),
                this.decoderBias1.read()
            )

            outputs = tf.layers
                .activation({ activation: this.decoderActivation })
                .apply(outputs)

            outputs = this.applyDense(
                outputs,
                this.decoderKernel2.read(),
                this.decoderBias2.read()
            )

            return outputs
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getWeights() {
        return [
            this.encoderKernel1.read(),
            this.encoderBias1.read(),
            this.encoderKernel2.read(),
            this.encoderBias2.read(),
            this.decoderKernel1.read(),
            this.decoderBias1.read(),
            this.decoderKernel2.read(),
            this.decoderBias2.read()
        ]
    }

    setWeights(weights) {
        this.encoderKernel1.write(weights[0])
        this.encoderBias1.write(weights[1])
        this.encoderKernel2.write(weights[2])
        this.encoderBias2.write(weights[3])
        this.decoderKernel1.write(weights[4])
        this.decoderBias1.write(weights[5])
        this.decoderKernel2.write(weights[6])
        this.decoderBias2.write(weights[7])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            innerDim: this.innerDim,
            bottleneck: this.bottleneck,
            outputDim: this.outputDim,
            encoderActivation: this.encoderActivation,
            decoderActivation: this.decoderActivation,
            variational: this.variational,
            beta: this.beta
        }
    }
}

class FastAssociativeMemory extends LayerBase {
    constructor(config) {
        super({ name: `mem-${randomString()}`, ...config })
        this.activation = config.activation || 'relu'
        this.steps = config.steps || 3
        this.learningRate = config.learningRate || 1e-3
        this.decayRate = config.decayRate || 0.9
        this.hPrev = null
        this.hHistory = []
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.W = this.addWeight(
            'W',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.C = this.addWeight(
            'C',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.b = this.addWeight(
            'b',
            [inputDim],
            'float32',
            tf.initializers.zeros()
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const seqLen = inputs.shape[1]

            if (!this.hPrev) {
                this.hPrev = tf.zerosLike(inputs)
                this.hHistory.push(tf.keep(this.hPrev.clone()))
            } else {
                const prevSeqLen = this.hPrev.shape[1]
                if (prevSeqLen < seqLen) {
                    const paddings = [
                        [0, 0],
                        [seqLen - prevSeqLen, 0],
                        [0, 0]
                    ]
                    this.hPrev = this.hPrev.pad(paddings, 1)
                    this.hHistory = this.hHistory.map((h) =>
                        tf.keep(h.pad(paddings, 1))
                    )
                } else if (prevSeqLen > seqLen) {
                    const paddings = [
                        [0, 0],
                        [prevSeqLen - seqLen, 0],
                        [0, 0]
                    ]
                    inputs = inputs.pad(paddings, 0)
                }
            }

            let hInitial = this.applyDense(inputs, this.C, this.b)
            hInitial = hInitial.add(this.applyDense(this.hPrev, this.W))

            hInitial = this.rmsNorm(hInitial)

            hInitial = tf.layers
                .activation({ activation: this.activation })
                .apply(hInitial)

            let h = hInitial
            for (let s = 0; s < this.steps; s++) {
                const attentionTerms = this.hHistory.map((hHist, idx) => {
                    const scalarProduct = tf.sum(tf.mul(hHist, h), -1, true)

                    const weightedProduct = tf.mul(
                        scalarProduct,
                        Math.pow(this.decayRate, this.hHistory.length - idx - 1)
                    )
                    return tf.mul(weightedProduct, hHist)
                })

                const attention = tf.sum(tf.stack(attentionTerms), 0)

                const hNext = tf.add(
                    hInitial,
                    tf.mul(attention, this.learningRate)
                )

                h = this.rmsNorm(hNext)

                h = tf.layers
                    .activation({ activation: this.activation })
                    .apply(h)
            }

            while (this.hHistory.length > this.steps) {
                this.hHistory[0].dispose()
                this.hHistory.shift()
            }

            this.hPrev = tf.keep(h)
            this.hHistory.push(tf.keep(h))

            return inputs.add(h)
        })
    }

    getWeights() {
        return [this.W.read(), this.C.read(), this.b.read()]
    }

    setWeights(weights) {
        this.W.write(weights[0])
        this.C.write(weights[1])
        this.b.write(weights[2])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            activation: this.activation,
            steps: this.steps,
            learningRate: this.learningRate,
            decayRate: this.decayRate
        }
    }
}

class OuroboticMemory extends LayerBase {
    constructor(config) {
        super({ name: `mem-${randomString()}`, ...config })
        this.steps = config.steps || 3
        this.decayRate = config.decayRate || 0.9
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.hPrev = null
        this.hHistory = []
        this.learningRate = []
        this.alpha = []
        this.activation = customActivations.Snake
        this.W = this.addWeight(
            'W',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.C = this.addWeight(
            'C',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.b = this.addWeight(
            'b',
            [inputDim],
            'float32',
            tf.initializers.zeros()
        )
        this.pAlpha = this.addWeight(
            `pAlpha`,
            [],
            'float32',
            tf.initializers.constant({ value: 0.2 }),
            tf.constraints.minMaxNorm({
                minValue: -1.0,
                maxValue: 1.0,
                rate: 0.9
            })
        )
        for (let i = 0; i < this.steps; i++) {
            this.learningRate.push(
                this.addWeight(
                    `learningRate-${i}`,
                    [],
                    'float32',
                    tf.initializers.constant({
                        value: 1e-7
                    }),
                    tf.constraints.minMaxNorm({
                        minValue: -0.001,
                        maxValue: 0.001,
                        rate: 0.99
                    })
                )
            )
            this.alpha.push(
                this.addWeight(
                    `alpha-${i}`,
                    [inputDim],
                    'float32',
                    tf.initializers.ones(),
                    tf.constraints.minMaxNorm({
                        minValue: 0.1,
                        maxValue: 23.0,
                        rate: 0.99
                    })
                )
            )
        }
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const seqLen = inputs.shape[1]

            if (!this.hPrev) {
                this.hPrev = tf.zerosLike(inputs)
                this.hHistory.push(tf.keep(this.hPrev.clone()))
            } else {
                const prevSeqLen = this.hPrev.shape[1]
                if (prevSeqLen < seqLen) {
                    const paddings = [
                        [0, 0],
                        [seqLen - prevSeqLen, 0],
                        [0, 0]
                    ]
                    this.hPrev = this.hPrev.pad(paddings, 1)
                    this.hHistory = this.hHistory.map((h) =>
                        tf.keep(h.pad(paddings, 1))
                    )
                } else if (prevSeqLen > seqLen) {
                    const paddings = [
                        [0, 0],
                        [prevSeqLen - seqLen, 0],
                        [0, 0]
                    ]
                    inputs = inputs.pad(paddings, 0)
                }
            }

            let hInitial = this.applyDense(inputs, this.C, this.b)

            hInitial = hInitial.add(this.applyDense(this.hPrev, this.W))

            hInitial = this.rmsNorm(hInitial).prelu(this.pAlpha.read())

            let h = hInitial
            for (let s = 0; s < this.steps; s++) {
                const attentionTerms = this.hHistory.map((hHist, idx) => {
                    const scalarProduct = tf.sum(tf.mul(hHist, h), -1, true)

                    const weightedProduct = tf.mul(
                        scalarProduct,
                        Math.pow(this.decayRate, this.hHistory.length - idx - 1)
                    )
                    return tf.mul(weightedProduct, hHist)
                })

                const attention = tf.sum(tf.stack(attentionTerms), 0)

                const hNext = tf.add(
                    hInitial,
                    tf.mul(attention, this.learningRate[s].read())
                )

                h = this.rmsNorm(hNext)

                h = this.activation.apply(h, this.alpha[s].read())
            }

            while (this.hHistory.length > this.steps) {
                this.hHistory[0].dispose()
                this.hHistory.shift()
            }

            this.hPrev = tf.keep(h)
            this.hHistory.push(tf.keep(h))

            return inputs.add(h)
        })
    }

    getWeights() {
        return [
            this.W.read(),
            this.C.read(),
            this.b.read(),
            this.pAlpha.read(),
            this.learningRate.map((weight) => weight.read()),
            this.alpha.map((weight) => weight.read())
        ].flat()
    }

    setWeights(weights) {
        this.W.write(weights[0])
        this.C.write(weights[1])
        this.b.write(weights[2])
        this.pAlpha.write(weights[3])
        for (let i = 4; i < this.steps; i++) {
            this.learningRate[i].write(weights[i])
        }
        for (let i = 4 + this.steps; i < this.steps; i++) {
            this.alpha[i].write(weights[i])
        }
    }

    getConfig() {
        return {
            ...super.getConfig(),
            steps: this.steps,
            decayRate: this.decayRate
        }
    }
}

// class Autoencoder extends LayerBase {
//     constructor(config) {
//         super({ name: `dia-${randomString()}`, ...config })
//         this.innerDim = config?.innerDim || 1024
//         this.bottleneck = config?.bottleneck || 128
//         this.outputDim = config?.outputDim || 256
//         this.encoderActivation = config?.encoderActivation || 'relu'
//         this.decoderActivation = config?.decoderActivation || 'relu'
//         this.variational = config?.variational || false
//         this.downsampling = config?.downsampling || {
//             strategy: 'train',
//             rate: 1.0
//         }
//         this.largestTimestep = 0
//     }

//     build(inputShape) {
//         const inputDim = inputShape[inputShape.length - 1]

//         // Initialize dense layers for encoder
//         this.encoderKernel1 = this.addWeight(
//             'encoderKernel1',
//             [inputDim, this.innerDim],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.encoderBias1 = this.addWeight(
//             'encoderBias1',
//             [this.innerDim],
//             'float32',
//             tf.initializers.zeros()
//         )
//         const multiplier = this.variational ? 2 : 1
//         this.encoderKernel2 = this.addWeight(
//             'encoderKernel2',
//             [this.innerDim, this.bottleneck * multiplier],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.encoderBias2 = this.addWeight(
//             'encoderBias2',
//             [this.bottleneck * multiplier],
//             'float32',
//             tf.initializers.zeros()
//         )

//         // Initialize dense layers for decoder
//         this.decoderKernel1 = this.addWeight(
//             'decoderKernel1',
//             [this.bottleneck, this.innerDim],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.decoderBias1 = this.addWeight(
//             'decoderBias1',
//             [this.innerDim],
//             'float32',
//             tf.initializers.zeros()
//         )
//         this.decoderKernel2 = this.addWeight(
//             'decoderKernel2',
//             [this.innerDim, this.outputDim],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.decoderBias2 = this.addWeight(
//             'decoderBias2',
//             [this.outputDim],
//             'float32',
//             tf.initializers.zeros()
//         )
//         if (this.downsampling.strategy === 'convolutional') {
//             this.reductionKernel = this.addWeight(
//                 'reductionKernel',
//                 [1, this.bottleneck, this.bottleneck],
//                 'float32',
//                 tf.initializers.glorotNormal()
//             )
//         }
//     }

//     computeVariance(inputs) {
//         // Split the encoded representation into mean and log-variance
//         const mean = inputs.slice([0, 0, 0], [-1, -1, this.bottleneck])
//         const logVar = inputs.slice(
//             [0, 0, this.bottleneck],
//             [-1, -1, this.bottleneck]
//         )

//         // Sample from the latent space using the reparameterization trick
//         const epsilon = tf.randomNormal(mean.shape)
//         return mean.add(epsilon.mul(logVar.exp().sqrt()))
//     }

//     // computeDownsampling(inputs) {
//     //     const inputTimesteps = inputs.shape[1]
//     //     const reducedTimesteps = Math.floor(
//     //         inputTimesteps / this.downsampling.rate
//     //     )

//     //     if (reducedTimesteps > this.largestTimestep) {
//     //         this.largestTimestep = reducedTimesteps
//     //     }

//     //     if (inputTimesteps <= this.largestTimestep) {
//     //         return inputs
//     //     }

//     //     // Apply timestep reduction by dropping tokens on the left
//     //     if (this.downsampling.strategy === 'truncate') {
//     //         return inputs.slice(
//     //             [0, inputTimesteps - reducedTimesteps, 0],
//     //             [-1, -1, -1]
//     //         )
//     //     }
//     //     // Apply timestep reduction via subsampling with trainable parameters
//     //     else if (this.downsampling.strategy === 'train') {
//     //         if (!this.trainableIndices) {
//     //             this.trainableIndices = this.addWeight(
//     //                 'trainableIndices',
//     //                 [inputTimesteps],
//     //                 'float32',
//     //                 tf.initializers.zeros()
//     //             )
//     //         }

//     //         // Use Gumbel-Softmax trick to select timesteps
//     //         const temperature = 1.0
//     //         const probabilities = customOps.gumbelSoftmax(
//     //             this.trainableIndices.read(),
//     //             temperature
//     //         )

//     //         // Select top-k timesteps based on probabilities
//     //         const topkValues = tf.topk(probabilities, reducedTimesteps)
//     //         const selectedIndices = topkValues.indices.arraySync()

//     //         // Use tf.gather to select the relevant timesteps
//     //         return tf.gather(inputs, selectedIndices, 1)
//     //     }
//     //     // Apply timestep reduction using FFT
//     //     else if (this.downsampling.strategy === 'ifft') {
//     //         return customOps.reduceTimeStepsWithFFT(inputs, reducedTimesteps)
//     //     } else if (this.downsampling.strategy === 'threshold') {
//     //         return customOps.reduceTimeStepsWithActivation(inputs, tf.tanh, 0.5)
//     //     } else if (this.downsampling.strategy === 'random') {
//     //         function keepValues(array, numToKeep) {
//     //             if (numToKeep >= array.length) return array

//     //             const remainingElements = array.slice() // Create a copy of the array

//     //             while (remainingElements.length > numToKeep) {
//     //                 let removalIndex = getRemovalIndex(remainingElements.length)
//     //                 remainingElements.splice(removalIndex, 1)
//     //             }

//     //             return remainingElements
//     //         }

//     //         function getRemovalIndex(length) {
//     //             let sumOfWeights = (length * (length + 1)) / 2 // Sum of the first N natural numbers
//     //             let randomValue = Math.random() * sumOfWeights

//     //             for (let i = 0; i < length; i++) {
//     //                 sumOfWeights -= length - i
//     //                 if (randomValue >= sumOfWeights) {
//     //                     return i
//     //                 }
//     //             }
//     //         }
//     //         const array = Array.from(
//     //             { length: reducedTimesteps },
//     //             (value, index) => index
//     //         )
//     //         const indices = keepValues(array, reducedTimesteps)
//     //         return tf.gather(inputs, indices, 1)
//     //     } else {
//     //         return inputs
//     //     }
//     // }

//     computeDownsampling(inputs) {
//         const factor = 2
//         return tf.conv1d(inputs, this.reductionKernel.read(), factor, 'valid')
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             // Encode the inputs to the bottleneck representation
//             let outputs = this.applyDense(
//                 inputs,
//                 this.encoderKernel1,
//                 this.encoderBias1
//             )

//             outputs = tf.layers
//                 .activation({ activation: this.encoderActivation })
//                 .apply(outputs)

//             outputs = this.applyDense(
//                 outputs,
//                 this.encoderKernel2,
//                 this.encoderBias2
//             )

//             if (this.variational) {
//                 outputs = this.computeVariance(outputs)
//             }

//             // if (this.downsampling.rate !== 1.0) {
//             //     outputs = this.computeDownsampling(outputs)
//             // }
//             outputs = this.computeDownsampling(outputs)

//             // Decode the bottleneck representation to the output dimensionality
//             outputs = this.applyDense(
//                 outputs,
//                 this.decoderKernel1,
//                 this.decoderBias1
//             )

//             outputs = tf.layers
//                 .activation({ activation: this.decoderActivation })
//                 .apply(outputs)

//             outputs = this.applyDense(
//                 outputs,
//                 this.decoderKernel2,
//                 this.decoderBias2
//             )

//             return outputs
//         })
//     }

//     computeOutputShape(inputShape) {
//         return [inputShape[0], inputShape[1], this.outputDim]
//     }

//     getWeights() {
//         const weights = [
//             this.encoderKernel1.read(),
//             this.encoderBias1.read(),
//             this.encoderKernel2.read(),
//             this.encoderBias2.read(),
//             this.decoderKernel1.read(),
//             this.decoderBias1.read(),
//             this.decoderKernel2.read(),
//             this.decoderBias2.read()
//         ]
//         // if (this.reductionKernel) {
//         //     weights.push(this.reductionKernel.read())
//         // }
//         return weights
//     }

//     setWeights(weights) {
//         this.encoderKernel1.write(weights[0])
//         this.encoderBias1.write(weights[1])
//         this.encoderKernel2.write(weights[2])
//         this.encoderBias2.write(weights[3])
//         this.decoderKernel1.write(weights[4])
//         this.decoderBias1.write(weights[5])
//         this.decoderKernel2.write(weights[6])
//         this.decoderBias2.write(weights[7])
//         // if (weights[8]) {
//         //     this.reductionKernel.write(weights[8])
//         // }
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             innerDim: this.innerDim,
//             bottleneck: this.bottleneck,
//             outputDim: this.outputDim,
//             encoderActivation: this.encoderActivation,
//             decoderActivation: this.decoderActivation,
//             variational: this.variational,
//             causalMasking: this.causalMasking,
//             downsampling: this.downsampling
//         }
//     }
// }

class CapsNet extends LayerBase {
    constructor(config) {
        super({ name: `cap-${randomString()}`, ...config })
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || 1e-5
        this.numCapsules = config?.numCapsules || 8
        this.capsuleDim = config?.capsuleDim || 16
        this.routingIterations = config?.routingIterations || 3
        this.activation = config?.activation || 'relu'
        this.supportsMasking = true
    }

    build(inputShape) {
        // Initialize dense layers for projection
        this.inProj = tf.layers.dense({
            units: this.innerDim,
            inputDim: this.units,
            activation: this.activation
        })
        this.outProj = tf.layers.dense({
            units: this.units,
            inputDim: this.capsuleDim * this.numCapsules,
            activation: 'linear'
        })

        // Initialize capsule layers
        this.primaryCaps = tf.layers.dense({
            units: this.numCapsules * this.capsuleDim,
            inputDim: this.innerDim,
            activation: 'linear'
        })
        this.digitCaps = new DigitCaps({
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            routingIterations: this.routingIterations
        })

        // Manually call build on layers to initialize weights
        this.inProj.build(inputShape)
        this.primaryCaps.build([inputShape[0], this.innerDim])
        this.digitCaps.build([inputShape[0], this.numCapsules, this.capsuleDim])
        this.outProj.build([inputShape[0], this.numCapsules * this.capsuleDim])

        // Initialize layer normalization
        this.layernorm = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layernorm.build(inputShape)

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand and contract projection via feedforward layers
            let outputs = this.inProj.apply(inputs)
            // Apply primary capsules
            outputs = this.primaryCaps.apply(outputs)
            outputs = tf.reshape(outputs, [
                -1,
                this.numCapsules,
                this.capsuleDim
            ])
            // Apply digit capsules with dynamic routing
            outputs = this.digitCaps.apply(outputs)
            outputs = tf.reshape(outputs, [
                -1,
                this.numCapsules * this.capsuleDim
            ])
            outputs = this.outProj.apply(outputs)
            // If training, apply residual dropout
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs
            outputs = tf.reshape(outputs, inputs.shape)
            // Apply layer norm
            outputs = this.layernorm.apply(outputs)
            // Apply skip connection
            return this.residual.apply([inputs, outputs])
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim
        }
    }
}

class DigitCaps extends LayerBase {
    constructor(config) {
        super({ name: `dcap-${randomString()}`, ...config })
        this.numCapsules = config.numCapsules
        this.capsuleDim = config.capsuleDim
        this.routingIterations = config.routingIterations || 3
    }

    build(inputShape) {
        this.W = this.addWeight(
            'capsules',
            [
                1,
                inputShape[1],
                this.numCapsules,
                this.capsuleDim,
                inputShape[2]
            ],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.built = true
    }

    call(inputs) {
        return tf.tidy(() => {
            const [batchSize, numPrimaryCaps, primaryCapsDim] = inputs.shape
            const uji = tf.tile(tf.expandDims(inputs, 2), [
                1,
                1,
                this.numCapsules,
                1
            ])
            const ujiHat = tf.sum(
                tf.mul(this.W.read(), tf.expandDims(uji, 3)),
                4
            )

            let b = tf.zeros([batchSize, numPrimaryCaps, this.numCapsules])
            let v = null
            for (let i = 0; i < this.routingIterations; i++) {
                const c = tf.softmax(b, 2)
                const s = tf.sum(tf.mul(tf.expandDims(c, 3), ujiHat), 1)
                v = this.squash(s)
                const agreement = tf.sum(tf.mul(ujiHat, tf.expandDims(v, 1)), 3)
                b = tf.add(b, agreement)
            }

            return v
        })
    }

    squash(s) {
        const squaredNorm = tf.sum(tf.square(s), -1, true)
        const squashFactor = tf.div(squaredNorm, tf.add(1, squaredNorm))
        return tf.mul(squashFactor, tf.div(s, tf.sqrt(squaredNorm)))
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.numCapsules, this.capsuleDim]
    }

    getConfig() {
        const config = {
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            routingIterations: this.routingIterations
        }
        return config
    }
}

class DebugLayer extends LayerBase {
    constructor(config) {
        super(config)
        this.config = config
        this.supportsMasking = true
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            console.log(inputs)
            inputs.print()
            console.log(inputs.dataSync())
            console.log(inputs.shape)
            return inputs
        })
    }
}

class Range extends LayerBase {
    constructor(config) {
        super({ name: `ran-${randomString()}`, ...config })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)
            const [B, T] = input.shape
            const range = tf.reshape(tf.range(0, T, 1, 'int32'), [1, T]) // .tile([B, 1])
            return range
        })
    }
}

class CausalSelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.config = Object.assign({ name: `attn-${randomString()}` }, config)

        // Config
        this.blockSize = config.blockSize || 256
        this.units = config.units || 256
        this.heads = config.heads || 4
        this.dropout = config.dropout || 0
        this.bias = config.bias || false
        this.epsilon = config.epsilon || 1e-5
        // Causal mask
        this.mask = tf.keep(
            tf.linalg.bandPart(
                tf.ones([config.blockSize, config.blockSize]),
                -1,
                0
            )
        )
    }

    build(inputShape) {
        this.cAttnKernel = this.addWeight(
            `c_attn-${randomString()}`,
            [this.units, 3 * this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.cAttnBias = this.addWeight(
            `c_attn-${randomString()}`,
            [3 * this.units],
            'float32',
            tf.initializers.zeros()
        )
        this.cProjKernel = this.addWeight(
            `c_proj-${randomString()}`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.cProjBias = this.addWeight(
            `c_proj-${randomString()}`,
            [this.units],
            'float32',
            tf.initializers.zeros()
        )
        this.layerNorm = tf.layers.layerNormalization({ epsilon: this.epsilon })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        // This is needed to save and load the model
        // When the model is saved, the config is saved with it
        // When the model is loaded, the config is used to create a new instance of the layer
        const config = super.getConfig()
        return Object.assign({}, config, this.config)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0]
            }

            const cAttn = this.applyDense(
                inputs,
                this.cAttnKernel,
                this.cAttnBias
            )

            // Make order of qkv split to follow minGPT
            let [q, k, v] = tf.split(cAttn, 3, -1)
            const [B, T, C] = k.shape

            const splitHeads = (x) =>
                tf.transpose(
                    tf.reshape(x, [B, T, this.heads, C / this.heads]),
                    [0, 2, 1, 3]
                )

            q = splitHeads(q)
            k = splitHeads(k)
            v = splitHeads(v)

            // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            let att = tf.mul(
                tf.matMul(q, k, false, true),
                tf.div(
                    1,
                    tf.sqrt(tf.cast(k.shape[k.shape.length - 1], 'float32'))
                )
            )

            const mask = this.mask.slice([0, 0], [T, T])
            att = tf.add(att, tf.mul(tf.sub(1, mask), -1e9))
            att = tf.softmax(att, -1)
            att = kwargs['training'] ? tf.dropout(att, this.dropout) : att

            let outputs = tf.matMul(att, v)

            outputs = tf.transpose(outputs, [0, 2, 1, 3])
            outputs = tf.reshape(outputs, [B, T, C])
            outputs = this.applyDense(outputs, this.cProjKernel, this.cProjBias)
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            outputs = this.layerNorm.apply(outputs)
            return tf.layers.add().apply([inputs, outputs])
        })
    }
}

class SinusoidalPositionalEncoding extends LayerBase {
    constructor(config) {
        super({ name: `enc-${randomString()}`, ...config })
        this.reverse = config?.reverse || false
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const range = customLayers.Range().apply(inputs)

            // Determine the sequence length from the input shape
            const seqLength = range.shape[1]

            // Compute the positional encodings (2D tensor of shape [seqLength, this.units])
            const positionalEncoding = tf.tensor2d(
                Array.from({ length: seqLength }, (_, pos) => {
                    return Array.from({ length: inputs.shape[2] }, (_, i) => {
                        const divTerm = Math.pow(
                            10000,
                            (2 * (i / 2)) / inputs.shape[2]
                        )
                        // Switch between sine and cosine based on the flag
                        if (this.reverse) {
                            return i % 2 === 0
                                ? Math.cos(pos / divTerm)
                                : Math.sin(pos / divTerm)
                        } else {
                            return i % 2 === 0
                                ? Math.sin(pos / divTerm)
                                : Math.cos(pos / divTerm)
                        }
                    })
                })
            )

            // Broadcast the positional encoding to match the shape of the inputs
            const broadcastedPositionalEncoding = positionalEncoding
                .expandDims(0)
                .tile([inputs.shape[0], 1, 1])

            return inputs.add(broadcastedPositionalEncoding)
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }
}

class MixtureOfDepthsRouting extends LayerBase {
    constructor(config) {
        super({ name: `mod-${randomString()}`, ...config })
        this.k = config.k
        this.units = config.units
    }

    build(inputShape) {
        this.routingWeights = this.addWeight(
            'routingWeights',
            [inputShape[inputShape.length - 1], 1],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const batchSize = inputs.shape[0]
            const seqLength = inputs.shape[1]

            const scores = tf.matMul(inputs, this.routingWeights.read())
            const reshapedScores = tf.reshape(scores, [batchSize, seqLength])
            const topkIndices = tf.topk(reshapedScores, this.k).indices

            const batchIndices = tf
                .range(0, batchSize)
                .expandDims(1)
                .tile([1, this.k])
                .cast('int32')
            const indices = tf.stack(
                [batchIndices.flatten(), topkIndices.flatten()],
                1
            )

            const selectedTokens = tf
                .gather(inputs.reshape([batchSize * seqLength, -1]), indices)
                .reshape([batchSize, this.k, -1])

            return selectedTokens
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            k: this.k,
            units: this.units
        }
    }
}

class LazyMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `lmoe-${randomString()}`, ...config })
        this.numRoutingIterations = config.numRoutingIterations || 3
        this.topK = config.topK || 2
        this.experts = config.experts
        this.numExperts = this.experts.length
        this.topK = Math.min(this.topK, this.numExperts)
    }

    build(inputShape) {
        this.experts.forEach((expert) => {
            expert.build(inputShape)
            // this._trainableWeights.push(...expert.trainableWeights)
        })
        super.build(inputShape)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const routingLogits = tf.variable(
                tf.zeros([inputs.shape[0], this.numExperts])
            )

            for (let i = 0; i < this.numRoutingIterations; i++) {
                const routingWeights = tf.softmax(routingLogits, -1)
                const topKIndices = tf.topk(routingWeights, this.topK).indices

                const expertOutputs = tf.stack(
                    topKIndices
                        .arraySync()
                        .map((indices) =>
                            tf.stack(
                                indices.map((index) =>
                                    this.experts[index].apply(inputs)
                                )
                            )
                        )
                )

                const routingWeightsGathered = tf.gather(
                    routingWeights,
                    topKIndices,
                    1
                )
                const combinedPredictions = tf.sum(
                    expertOutputs.mul(
                        routingWeightsGathered.expandDims(-1).expandDims(-1)
                    ),
                    1
                )

                const agreementScores = tf.sum(
                    expertOutputs.mul(combinedPredictions.expandDims(1)),
                    [-1, -2]
                )

                const batchIndices = tf.range(0, inputs.shape[0])
                const flattenedTopKIndices = topKIndices.flatten()

                const updates = agreementScores.reshape([
                    inputs.shape[0],
                    this.topK
                ])
                console.log(updates)

                const indices = tf.stack(
                    [
                        batchIndices.cast('int32'),
                        flattenedTopKIndices.cast('int32')
                    ],
                    1
                )

                routingLogits.assign(
                    routingLogits.add(
                        tf.scatterND(indices, updates, routingLogits.shape)
                    )
                )
            }

            const routingWeights = tf.softmax(routingLogits, -1)
            const topKIndices = tf.topk(routingWeights, this.topK).indices

            const expertOutputs = tf.stack(
                topKIndices
                    .arraySync()
                    .map((indices) =>
                        tf.stack(
                            indices.map((index) =>
                                this.experts[index].apply(inputs)
                            )
                        )
                    )
            )

            const routingWeightsGathered = tf.gather(
                routingWeights,
                topKIndices,
                1
            )
            const finalOutput = tf.sum(
                expertOutputs.mul(
                    routingWeightsGathered.expandDims(-1).expandDims(-1)
                ),
                1
            )

            return finalOutput
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            numRoutingIterations: this.numRoutingIterations,
            topK: this.topK
        }
    }
}

class ControlGate extends LayerBase {
    constructor(config) {
        super({ name: `gate-${randomString()}`, ...config })
        this.units = config.units || 64
        this.experts = config.experts // Array of expert layers
        this.numExperts = this.experts.length
        this.currentBatch
        this.currentExperts = []
    }

    build(inputShape) {
        this.in_gate = tf.layers.dense({
            name: `gate-${randomString()}`,
            units: this.units,
            activation: 'swish'
        })

        this.out_gate = tf.layers.dense({
            name: `gate-${randomString()}`,
            units: this.numExperts,
            activation: 'softmax'
        })

        // Build gating mechanism
        this.in_gate.build(inputShape)
        let gateOutputShape = this.in_gate.computeOutputShape(inputShape)
        this.out_gate.build(gateOutputShape)

        // this._trainableWeights = [
        //     ...this.in_gate.trainableWeights,
        //     ...this.out_gate.trainableWeights
        // ]

        // Build each expert layer
        this.experts.forEach((expert) => {
            expert.build(inputShape)
            // this._trainableWeights.push(...expert.trainableWeights)
        })

        super.build(inputShape)
    }

    computeGate(inputs) {
        return this.out_gate.apply(this.in_gate.apply(inputs))
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // We always train the gate, even if we do not use its output 100% of the time
            const gatingScores = this.computeGate(inputs)

            // Compute outputs sequentially over the first probability distribution
            if (
                !kwargs.training ||
                this.currentBatch !== kwargs.batch ||
                this.currentExperts.length === 0
            ) {
                this.currentBatch = kwargs.batch
                const topKValues = tf.topk(gatingScores, this.numExperts, true)
                this.currentExperts = topKValues.indices
                    .slice([0, inputs.shape[1] - 1, 0], [1, 1, this.numExperts])
                    .reshape([this.numExperts])
                    .arraySync()
            }

            // Take the highest probability expert
            const choice = this.currentExperts.shift()
            return this.experts[choice].apply(inputs)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            numExperts: this.numExperts
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

// Originally adapted from:
// https://gist.githubusercontent.com/BenjaminWegener/311292080a71becbe5a8c0cc7657657d/raw/fd4f1f96184b58dace1854d0440d8c9dde3fd712/attention_layer_tfjs
class LambdaLayer extends LayerBase {
    constructor(config) {
        super(config)
        if (config.name === undefined) {
            config.name = (+new Date() * Math.random()).toString(36)
        }
        this.name = config.name
        this.lambdaFunction = config.lambdaFunction
        this.lambdaOutputShape = config.lambdaOutputShape
    }
    call(input) {
        return tf.tidy(() => {
            let result = null
            eval(this.lambdaFunction)
            return result
        })
    }
    computeOutputShape(inputShape) {
        if (this.lambdaOutputShape === undefined) {
            //if no outputshape provided, try to set as inputshape
            return inputShape[0]
        } else {
            return this.lambdaOutputShape
        }
    }
    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            lambdaFunction: this.lambdaFunction,
            lambdaOutputShape: this.lambdaOutputShape
        })
        return config
    }
}

// https://arxiv.org/abs/2005.00743
// https://github.com/iafarhan/causal-synthesizer-multihead-attention/blob/main/synthesizer.py
class SynthesizerAttention extends LayerBase {
    constructor(config) {
        super({ name: `syn-${randomString()}`, ...config })
        this.units = config.units
        this.heads = config.heads
        this.blockSize = config.blockSize
        this.attnPdrop = config.dropout || 0.0
        this.residPdrop = config.dropout || 0.0
        this.activation = tf.leakyRelu
        this.epsilon = config.epsilon || false
        this.alpha = config.alpha || 1
        this.depth = this.units / this.heads
    }

    build(inputShape) {
        this.w1 = this.addWeight(
            `w1`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.w2 = this.addWeight(
            `w2`,
            [this.units / this.heads, this.blockSize],
            'float32',
            tf.initializers.zeros()
        )
        this.b2 = this.addWeight(
            `b2`,
            [this.blockSize],
            'float32',
            tf.initializers.zeros()
        )
        this.value = this.addWeight(
            `value`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.proj = this.addWeight(
            `proj`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )

        if (this.epsilon) {
            this.layernorm = tf.layers.layerNormalization({
                epsilon: this.epsilon
            })
        }

        this.residual = new ResidualConnection()

        this.attnDropout = tf.layers.dropout({ rate: this.attnPdrop })
        this.residDropout = tf.layers.dropout({ rate: this.residPdrop })

        const mask = tf.linalg.bandPart(
            tf.ones([this.blockSize, this.blockSize]),
            -1,
            0
        )
        this.mask = tf.keep(tf.expandDims(tf.expandDims(mask, 0), 0))
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, seqLen, embedSize] = inputs.shape

            const nonlinearOut = this.activation(
                this.synthesize(inputs, this.w1.read()),
                this.alpha
            )
            const nonlinearReshaped = tf.transpose(
                tf.reshape(nonlinearOut, [
                    batchSize,
                    seqLen,
                    this.heads,
                    this.depth
                ]),
                [0, 2, 1, 3]
            )

            const v = this.synthesize(inputs, this.value.read())
            const vReshaped = tf.transpose(
                tf.reshape(v, [batchSize, seqLen, this.heads, this.depth]),
                [0, 2, 1, 3]
            )

            const w2Tiled = this.w2
                .read()
                .expandDims(0)
                .tile([batchSize * this.heads, 1, 1])
            let scores = tf.add(
                tf.reshape(
                    tf.matMul(
                        tf.reshape(nonlinearReshaped, [
                            batchSize * this.heads,
                            seqLen,
                            this.depth
                        ]),
                        w2Tiled
                    ),
                    [batchSize, this.heads, seqLen, this.blockSize]
                ),
                this.b2.read()
            )

            scores = scores.slice(
                [0, 0, 0, 0],
                [batchSize, this.heads, seqLen, seqLen]
            )

            scores = tf.where(
                tf.equal(
                    this.mask.slice([0, 0, 0, 0], [1, 1, seqLen, seqLen]),
                    0
                ),
                tf.fill([batchSize, this.heads, seqLen, seqLen], -1e10),
                scores
            )

            const probAttn = tf.softmax(scores, -1)
            const attnOutput = this.attnDropout.apply(probAttn)
            const y = tf.matMul(attnOutput, vReshaped)

            const yTransposed = tf.transpose(y, [0, 2, 1, 3])
            const yReshaped = tf.reshape(yTransposed, [
                batchSize,
                seqLen,
                embedSize
            ])

            let output = this.synthesize(yReshaped, this.proj.read())

            output = this.residDropout.apply(output)
            if (this.layernorm) output = this.layernorm.apply(output)

            return this.residual.apply([inputs, output])
        })
    }

    synthesize(x, kernel) {
        const k = kernel.expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.blockSize, this.units]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            units: this.units,
            heads: this.heads,
            blockSize: this.blockSize,
            attnPdrop: this.dropout,
            residPdrop: this.dropout,
            epsilon: this.epsilon,
            alpha: this.alpha,
            depth: this.depth
        })
        return config
    }
}

class SparseEstimatedAttention extends LayerBase {
    constructor(config) {
        super({ name: `sea-${randomString()}`, ...config })
        this.units = config.units
        this.heads = config.heads
        this.attnPdrop = config.attnPdrop || 0.0
        this.residPdrop = config.residPdrop || 0.0
        this.activation = config.activation || tf.leakyRelu
        this.alpha = config.alpha || 0.2
        this.depth = this.units / this.heads
        this.topK = config.topK || null
    }

    build(inputShape) {
        this.linearProj = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            useBias: false
        })

        this.value = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            useBias: false
        })

        this.proj = tf.layers.dense({
            units: inputShape[2],
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            useBias: false
        })

        this.kernel = tf.layers.dense({
            units: this.heads * this.depth,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            useBias: false
        })

        this.attnDropout = tf.layers.dropout({ rate: this.attnPdrop })
        this.residDropout = tf.layers.dropout({ rate: this.residPdrop })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, seqLen, embedSize] = inputs.shape

            const linearOut = this.linearProj.apply(inputs)
            const nonlinearOut = this.activation(linearOut, this.alpha)
            const nonlinearReshaped = tf.transpose(
                tf.reshape(nonlinearOut, [
                    batchSize,
                    seqLen,
                    this.heads,
                    this.depth
                ]),
                [0, 2, 1, 3]
            )

            const v = this.value.apply(inputs)
            const vReshaped = tf.transpose(
                tf.reshape(v, [batchSize, seqLen, this.heads, this.depth]),
                [0, 2, 1, 3]
            )

            const kernelOut = this.kernel.apply(inputs)
            const kernelReshaped = tf.reshape(kernelOut, [
                batchSize,
                seqLen,
                this.heads,
                this.depth
            ])

            const attnScores = tf.matMul(
                tf.reshape(nonlinearReshaped, [
                    batchSize * this.heads,
                    seqLen,
                    this.depth
                ]),
                tf.reshape(kernelReshaped, [
                    batchSize * this.heads,
                    this.depth,
                    seqLen
                ])
            )
            const attnMatrix = tf.reshape(attnScores, [
                batchSize,
                this.heads,
                seqLen,
                seqLen
            ])

            let sparseMask
            if (this.topK) {
                const { values, indices } = customOps.subliminalTopk(
                    attnMatrix,
                    this.topK
                )
                sparseMask = tf.oneHot(indices, seqLen).sum(3)
                indices.dispose()
            } else {
                sparseMask = tf.ones([batchSize, this.heads, seqLen, seqLen])
            }

            const maskedAttn = tf.mul(attnMatrix, sparseMask)
            const probAttn = tf.softmax(maskedAttn, -1)
            const attnOutput = this.attnDropout.apply(probAttn)

            const context = tf.matMul(attnOutput, vReshaped)

            const contextTransposed = tf.transpose(context, [0, 2, 1, 3])
            const contextFlattened = tf.reshape(contextTransposed, [
                batchSize,
                seqLen,
                this.units
            ])

            let output = this.proj.apply(contextFlattened)
            output = this.residDropout.apply(output)

            return output
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            heads: this.heads,
            attnPdrop: this.attnPdrop,
            residPdrop: this.residPdrop,
            activation: this.activation,
            alpha: this.alpha,
            depth: this.depth,
            topK: this.topK
        }
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

// class RotaryPositionalEncoding extends LayerBase {
//     constructor(config) {
//         super({ name: `rot-${randomString()}`, ...config })
//         this.blockSize = config.blockSize
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             const batchSize = inputs.shape[0]
//             const seqLen = inputs.shape[1]
//             const embeddingDim = inputs.shape[2]
//             const paddedInputs = inputs.pad([
//                 [0, 0],
//                 [0, Math.max(this.blockSize - seqLen, 0)],
//                 [0, 0]
//             ])
//             const paddedSeqLen = paddedInputs.shape[1]
//             const posEncoding = this.getRotaryPositionalEmbedding(
//                 paddedSeqLen,
//                 embeddingDim
//             )
//             const output = this.applyRotaryPositionalEmbedding(
//                 paddedInputs,
//                 posEncoding
//             )
//             return output.slice(
//                 [0, 0, 0],
//                 [batchSize, this.blockSize, embeddingDim]
//             )
//         })
//     }

//     getRotaryPositionalEmbedding(seqLen, embeddingDim) {
//         const pos = tf.range(0, seqLen, 1, 'float32').reshape([-1, 1])
//         const i = tf.range(0, embeddingDim / 2, 1, 'float32').reshape([1, -1])
//         const angleRates = tf.pow(10000, tf.div(i, embeddingDim / 2))
//         const angleRads = tf.mul(pos, tf.div(1, angleRates))
//         const sin = tf.sin(angleRads)
//         const cos = tf.cos(angleRads)

//         // Interleave sin and cos values
//         const sinExpanded = sin.expandDims(2) // Expanding dimension to enable interleaving
//         const cosExpanded = cos.expandDims(2)
//         const concatenated = tf.concat([sinExpanded, cosExpanded], 2) // Concatenate along the new axis
//         const posEncoding = concatenated.reshape([seqLen, embeddingDim])
//         return posEncoding
//     }

//     applyRotaryPositionalEmbedding(x, posEncoding) {
//         const embeddingDim = x.shape[2]
//         const xDtype = x.dtype

//         // Split the embedding dimension into two halves for sin and cos applications
//         const rotaryDim = embeddingDim / 2
//         const [xRot, xPass] = tf.split(x, 2, -1)

//         // Apply sin to the first half and cos to the second half of posEncoding
//         const sinEncoding = posEncoding.slice([0, 0], [-1, rotaryDim])
//         const cosEncoding = posEncoding.slice([0, rotaryDim], [-1, -1])

//         // Apply the encodings
//         const xRotSin = tf.mul(xRot, sinEncoding)
//         const xRotCos = tf.mul(xRot, cosEncoding)

//         // Reconstruct the rotated embeddings
//         const rotatedX = tf.concat([xRotSin, xRotCos], -1)

//         // Concatenate the rotated part with the part that does not get rotated
//         const output = tf.concat([rotatedX, xPass], -1)

//         return output.asType(xDtype)
//     }

//     computeOutputShape(inputShape) {
//         return [inputShape[0], this.blockSize, inputShape[2]]
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             blockSize: this.blockSize
//         }
//     }
// }

class CompressorHead extends LayerBase {
    constructor(config) {
        super({ name: `compressor-${randomString()}`, ...config })
        this.operations = config.operations || 3
        this.compressionFactor = config.compressionFactor || 2
        this.epsilon = config.epsilon || 1e-8
        this.weightVectors = []
        this.defaultOps = [
            'add',
            'sub',
            'min',
            'max',
            'leakyRelu',
            'tanh',
            'softplus',
            'sin',
            'cos'
        ]
        this.allowedOps = config.allowedOps ||
            this.defaultOps || [
                'add',
                'cos',
                'div',
                'leakyRelu',
                'max',
                'mean',
                'min',
                'mul',
                'norm',
                'relu',
                'sigmoid',
                'sin',
                'softplus',
                'sub',
                'tanh'
            ]
    }

    build(inputShape) {
        for (let i = 0; i < this.operations; i++) {
            const weightVector = this.addWeight(
                `head-${randomString()}`,
                [inputShape[2]],
                'float32',
                tf.initializers.glorotUniform()
            )
            this.weightVectors.push(weightVector)
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            this.setMode()

            const [batchSize, seqLen, embedDim] = inputs.shape

            // Decide operation based on mode
            if (this.mode === 'compress') {
                return this.compress(inputs, seqLen, embedDim, batchSize)
            } else if (this.mode === 'decompress') {
                return this.decompress(inputs, seqLen, embedDim, batchSize)
            }
        })
    }

    setMode() {
        if (this.mode === 'compress') {
            this.mode = 'decompress'
        } else {
            this.mode = 'compress'
        }
    }

    randomOperation(a, b, seed) {
        const op = seededValueFromArray(this.allowedOps, seed)

        let result
        if (op === 'add') {
            result = a.add(b)
        } else if (op === 'sub') {
            result = a.sub(b)
        } else if (op === 'mul') {
            result = a.mul(b)
        } else if (op === 'div') {
            result = a.div(b.add(this.epsilon))
        } else if (op === 'max') {
            result = a.maximum(b)
        } else if (op === 'min') {
            result = a.minimum(b)
        } else if (op === 'mean') {
            result = tf.mean(tf.stack([a, b]), 0)
        } else if (op === 'norm') {
            result = tf.norm(tf.stack([a, b]), 2, 0)
        } else if (op === 'relu') {
            result = tf.relu(a.add(b))
        } else if (op === 'leakyRelu') {
            result = tf.relu(a.sub(b))
        } else if (op === 'sigmoid') {
            result = tf.sigmoid(a.mul(b))
        } else if (op === 'tanh') {
            result = tf.tanh(a.div(b.add(this.epsilon)))
        } else if (op === 'softplus') {
            result = tf.softplus(a.add(b))
        } else if (op === 'sin') {
            result = tf.sin(a.add(b))
        } else if (op === 'cos') {
            result = tf.cos(a.sub(b))
        }
        // const mean = result.mean([-1], true);
        // const variance = result.sub(mean).square().mean([-1], true);
        // return result.sub(mean).div(variance.add(epsilon).sqrt());
        return result
    }

    // TODO: We should probably try to implement Principal Component Analysis (PCA) for dimensionality reduction here
    compress(inputs, seqLen, embedDim, batchSize) {
        const paddedSeqLen =
            Math.ceil(seqLen / this.compressionFactor) * this.compressionFactor
        const paddedInputs = inputs.pad([
            [0, 0],
            [0, paddedSeqLen - seqLen],
            [0, 0]
        ])

        const reshapedInputs = paddedInputs.reshape([
            batchSize,
            paddedSeqLen / this.compressionFactor,
            this.compressionFactor,
            embedDim
        ])

        const pooledVectors = this.weightVectors.map((weightVector) => {
            const expandedVector = weightVector
                .read()
                .expandDims(0)
                .expandDims(0)
                .expandDims(0)
            return tf.sum(reshapedInputs.mul(expandedVector), 2)
        })

        return pooledVectors.reduce((a, b, i) => {
            return this.randomOperation(a, b, i)
        })
    }

    decompress(inputs, seqLen, embedDim, batchSize) {
        const reshapedInputs = inputs
            .reshape([batchSize, seqLen, 1, embedDim])
            .tile([1, 1, this.compressionFactor, 1])
            .reshape([batchSize, seqLen * this.compressionFactor, 1, embedDim])

        const pooledVectors = this.weightVectors.map((weightVector) => {
            const expandedVector = weightVector
                .read()
                .expandDims(0)
                .expandDims(0)
                .expandDims(0)
            return tf.sum(reshapedInputs.mul(expandedVector), 2) // Sum over the synthetic 'compressionFactor' dimension
        })

        return pooledVectors.reduce((a, b, i) => {
            return this.randomOperation(a, b, seededPRNG(i))
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return Object.assign(super.getConfig(), {
            compressionFactor: this.compressionFactor,
            operations: this.operations
        })
    }
}

class DumbCompression extends LayerBase {
    constructor(config) {
        super(config)
        this.compressionFactor = config.compressionFactor
        this.poolingType = config.poolingType || 'avg'
    }

    build(inputShape) {
        if (this.poolingType !== 'dot') return
        const numVectors = 23
        this.weightVectors = []
        for (let i = 0; i < numVectors; i++) {
            const weightVector = this.addWeight(
                `weightVector${i}`,
                [inputShape[2]],
                'float32',
                tf.initializers.glorotUniform()
            )
            this.weightVectors.push(weightVector)
        }
    }

    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        const [batchSize, seqLen, embedDim] = inputs.shape

        const paddedSeqLen =
            Math.ceil(seqLen / this.compressionFactor) * this.compressionFactor
        const paddedInputs = inputs.pad([
            [0, 0],
            [0, paddedSeqLen - seqLen],
            [0, 0]
        ])

        const reshapedInputs = paddedInputs.reshape([
            batchSize,
            paddedSeqLen / this.compressionFactor,
            this.compressionFactor,
            embedDim
        ])

        let pooledEmbeddings
        if (this.poolingType === 'avg') {
            pooledEmbeddings = tf.mean(reshapedInputs, 2)
        } else if (this.poolingType === 'max') {
            pooledEmbeddings = tf.max(reshapedInputs, 2)
        } else if (this.poolingType === 'norm') {
            pooledEmbeddings = tf.norm(reshapedInputs, 2)
        } else if (this.poolingType === 'dot') {
            const pooledVectors = this.weightVectors.map((weightVector) => {
                const expandedVector = weightVector
                    .read()
                    .expandDims(0)
                    .expandDims(0)
                    .expandDims(0)
                return tf.sum(reshapedInputs.mul(expandedVector), 2)
            })
            // Combine the pooled vectors (e.g., sum, multiply, subtract, divide)
            pooledEmbeddings = pooledVectors.reduce((a, b, i) => {
                if (i % 2 === 0) {
                    return a.sub(b)
                } else {
                    return a.add(b)
                }
            })
        } else {
            throw new Error(`Unsupported pooling type: ${this.poolingType}`)
        }
        return pooledEmbeddings
    }

    computeOutputShape(inputShape) {
        const seqLen = Math.ceil(inputShape[1] / this.compressionFactor)
        return [inputShape[0], seqLen, inputShape[2]]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            compressionFactor: this.compressionFactor,
            poolingType: this.poolingType
        })
        return config
    }
}

class ConvolutionalExpansionLayer extends LayerBase {
    constructor(config) {
        super(config)
        this.seqLen = config.seqLen
        this.units = config.units // The desired depth of the output, similar to filters in conv2dTranspose
        this.kernelSize = config.kernelSize || 3 // Kernel size for the transposed convolution
    }

    build(inputShape) {
        const adjustedShape = [1, inputShape[1], inputShape[2]]
        // Define a 2D transposed convolution layer to simulate 1D sequence expansion
        this.conv2dTransposeLayer = tf.layers.conv2dTranspose({
            filters: this.units,
            kernelSize: [1, this.kernelSize], // Faux height of 1, actual kernel size for width
            strides: [1, 2], // Stride 2 for expansion, faux height stride of 1
            padding: 'same',
            activation: 'linear' // Consider the appropriate activation for your task
            // inputShape: [1, inputShape[1], inputShape[2]] // Adjusted for conv2d input
        })
        // this.conv2dTransposeLayer.build([null, 1, inputShape[1], inputShape[2]])
        this.conv2dTransposeLayer.build([null, ...adjustedShape])
        // this._trainableWeights.push(...this.conv2dTransposeLayer)

        super.build(inputShape) // Ensure the layer is marked as built
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            let input = inputs
            if (!Array.isArray(input)) {
                input = [input]
            }
            // Reshape the input to add a faux height dimension
            const reshapedInput = input[0].reshape([
                -1,
                1,
                input[0].shape[1],
                input[0].shape[2]
            ])

            // Apply the transposed convolution layer
            let output = this.conv2dTransposeLayer.apply(reshapedInput, kwargs)

            // Squeeze to remove the faux height dimension and possibly adjust width (sequence length)
            output = output.squeeze([1])

            // If necessary, trim or pad the output to exactly match the target sequence length
            const currentSeqLen = output.shape[1]
            if (currentSeqLen > this.seqLen) {
                // Trim excess length
                output = output.slice([0, 0, 0], [-1, this.seqLen, -1])
            } else if (currentSeqLen < this.seqLen) {
                // Pad to target length
                const padWidth = this.seqLen - currentSeqLen
                output = tf.pad(output, [
                    [0, 0],
                    [0, padWidth],
                    [0, 0]
                ])
            }

            return output
        })
    }

    computeOutputShape(inputShape) {
        // Assuming the transformation maintains or adjusts to the target sequence length
        return [inputShape[0], this.seqLen, this.units]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            seqLen: this.seqLen,
            units: this.units,
            kernelSize: this.kernelSize
        })
        return config
    }
}

class SequenceExpansionLayer extends LayerBase {
    constructor(config) {
        super(config)
        this.seqLen = config.seqLen
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = inputs instanceof Array ? inputs[0] : inputs
            const [batchSize, seqLen, embedDim] = input.shape

            // Check if expansion is needed
            if (seqLen >= this.seqLen) {
                return input
            }

            // Calculate the expansion factor based on current and target sequence lengths
            const expansionFactor = this.seqLen / seqLen

            // Create an expanded sequence by repeating elements (simplistic approach)
            let expandedInputs = input.tile([1, expansionFactor, 1])

            // Slice the expanded sequence to match the exact target sequence length
            let slicedExpandedInputs = expandedInputs.slice(
                [0, 0, 0],
                [batchSize, this.seqLen, embedDim]
            )

            return slicedExpandedInputs
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.seqLen, inputShape[2]]
    }
}

class NearestNeighborUpsampling extends LayerBase {
    constructor(config) {
        super(config)
        this.upsamplingFactor = config.upsamplingFactor
    }

    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        const [batchSize, seqLen, embedDim] = inputs.shape

        const upsampledSeqLen = seqLen * this.upsamplingFactor

        const reshapedInputs = inputs.reshape([batchSize, seqLen, 1, embedDim])

        const upsampledInputs = tf.tile(reshapedInputs, [
            1,
            1,
            this.upsamplingFactor,
            1
        ])

        const outputShape = [batchSize, upsampledSeqLen, embedDim]
        const output = upsampledInputs.reshape(outputShape)

        return output
    }

    computeOutputShape(inputShape) {
        const [batchSize, seqLen, embedDim] = inputShape
        const upsampledSeqLen = seqLen * this.upsamplingFactor
        return [batchSize, upsampledSeqLen, embedDim]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            upsamplingFactor: this.upsamplingFactor
        })
        return config
    }
}

class LearnedUpsampling extends LayerBase {
    constructor(config) {
        super(config)
        this.upsamplingFactor = config.upsamplingFactor
        this.filters = config.filters || config.units
        this.kernelSize = config.kernelSize || 3
        this.strides = config.strides || this.upsamplingFactor
        this.padding = config.padding || 'same'
        this.activation = config.activation || 'linear'
    }

    build(inputShape) {
        const [batchSize, seqLen, embedDim] = inputShape
        const kernelShape = [this.kernelSize, 1, this.filters, embedDim]
        this.kernel = this.addWeight(
            'kernel',
            kernelShape,
            'float32',
            tf.initializers.glorotUniform()
        )
        const biasShape = [this.filters]
        this.bias = this.addWeight(
            'bias',
            biasShape,
            'float32',
            tf.initializers.zeros()
        )
    }

    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        const inputShape = inputs.shape
        const [batchSize, seqLen, embedDim] = inputShape
        const reshapedInputs = inputs.reshape([batchSize, seqLen, 1, embedDim])
        const upsampledInputs = tf.conv2dTranspose(
            reshapedInputs,
            this.kernel.read(),
            [batchSize, seqLen * this.upsamplingFactor, 1, this.filters],
            [this.strides, 1],
            this.padding
        )
        const biasedOutput = tf.add(upsampledInputs, this.bias.read())
        const activatedOutput = tf.layers
            .activation({ activation: this.activation })
            .apply(biasedOutput)
        const outputShape = [
            batchSize,
            seqLen * this.upsamplingFactor,
            this.filters
        ]
        const output = activatedOutput.reshape(outputShape)
        return output
    }

    computeOutputShape(inputShape) {
        const [batchSize, seqLen, embedDim] = inputShape
        const outputSeqLen = seqLen * this.upsamplingFactor
        return [batchSize, outputSeqLen, this.filters]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            upsamplingFactor: this.upsamplingFactor,
            filters: this.filters,
            kernelSize: this.kernelSize,
            strides: this.strides,
            padding: this.padding,
            activation: this.activation
        })
        return config
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

// class StateSpace extends LayerBase {
//     constructor(config) {
//         super({ name: `ssm-${randomString()}`, ...config })
//         this.units = config.units || 64
//         this.innerDim = config.innerDim || 256
//         this.returnSequences = config.returnSequences || false
//         this.decayFactor = config.decayFactor || 1.0
//         this.activation = config.activation || 'tanh'
//     }

//     build(inputShape) {
//         const inputDim = inputShape[2]
//         this.kernel = this.addWeight(
//             'kernel',
//             [inputDim, this.innerDim],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.recurrentKernel = this.addWeight(
//             'recurrentKernel',
//             [this.units, this.innerDim],
//             'float32',
//             tf.initializers.orthogonal({ gain: 1 })
//         )
//         this.outputKernel = this.addWeight(
//             'outputKernel',
//             [this.innerDim, this.units],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.bias = this.addWeight(
//             'bias',
//             [this.innerDim],
//             'float32',
//             tf.initializers.zeros()
//         )

//         this.residual = new ResidualConnection()
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             const [batchSize, sequenceLength, inputDim] = inputs.shape

//             let state = tf.zeros([batchSize, this.units])
//             const outputs = []

//             const kernel = this.kernel.read()
//             const recurrentKernel = this.recurrentKernel.read()
//             const outputKernel = this.outputKernel.read()
//             const bias = this.bias.read()

//             for (let t = 0; t < sequenceLength; t++) {
//                 const input = inputs
//                     .slice([0, t, 0], [batchSize, 1, inputDim])
//                     .reshape([batchSize, inputDim])
//                 const innerState = tf
//                     .add(
//                         tf.matMul(input, kernel),
//                         tf.matMul(state, recurrentKernel).mul(this.decayFactor)
//                     )
//                     .add(bias)
//                 const activatedState = tf.layers
//                     .activation({ activation: this.activation })
//                     .apply(innerState)

//                 const newState = tf.matMul(activatedState, outputKernel)
//                 outputs.push(newState)
//                 state = newState
//             }

//             let output = this.returnSequences
//                 ? tf.stack(outputs, 1)
//                 : outputs[outputs.length - 1]

//             output = this.rmsNorm(output)

//             return this.residual.apply([inputs, output])
//         })
//     }

//     computeOutputShape(inputShape) {
//         const outputShape = this.returnSequences
//             ? [inputShape[0], inputShape[1], this.units]
//             : [inputShape[0], this.units]
//         return outputShape
//     }

//     getWeights() {
//         return [
//             this.kernel.read(),
//             this.outputKernel.read(),
//             this.recurrentKernel.read(),
//             this.bias.read()
//         ]
//     }

//     setWeights(weights) {
//         this.kernel.write(weights[0])
//         this.outputKernel.write(weights[1])
//         this.recurrentKernel.write(weights[2])
//         this.bias.write(weights[3])
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             units: this.units,
//             innerDim: this.innerDim,
//             returnSequences: this.returnSequences,
//             decayFactor: this.decayFactor,
//             activation: this.activation
//         }
//     }
// }

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

class StatePlacement extends LayerBase {
    constructor(config) {
        super({ name: `ssm-${randomString()}`, ...config })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.epsilon = config.epsilon || false
        this.k = config.k || 4
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
        this.attentionKernel = this.addWeight(
            'attentionKernel',
            [inputDim, 1],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.bias = this.addWeight(
            'bias',
            [this.innerDim],
            'float32',
            tf.initializers.zeros()
        )

        if (this.epsilon) {
            this.layernorm = tf.layers.layerNormalization({
                epsilon: this.epsilon
            })
        }

        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, sequenceLength, inputDim] = inputs.shape

            let state = tf.zeros([batchSize, this.units])

            const attentionScores = this.dense(
                inputs,
                this.attentionKernel,
                null
            ).reshape([batchSize, sequenceLength])
            const topkIndices = tf.topk(attentionScores, this.k).indices

            const selectedInputs = tf.gather(inputs, topkIndices, 1)
            const outputs = []

            for (let t = 0; t < this.k; t++) {
                const input = selectedInputs.slice(
                    [0, t, 0],
                    [batchSize, 1, inputDim]
                )
                const innerState = tf.tanh(
                    tf.add(
                        tf.add(
                            this.dense(
                                input.reshape([batchSize, inputDim]),
                                this.kernel,
                                this.bias
                            ),
                            this.dense(state, this.recurrentKernel, null)
                        ),
                        this.bias.read()
                    )
                )
                const newState = this.dense(innerState, this.outputKernel, null)
                outputs.push(newState)
                state = newState
            }

            let output = this.returnSequences
                ? tf.stack(outputs, 1)
                : outputs[outputs.length - 1]

            if (this.layernorm) output = this.layernorm.apply(output)

            return this.residual.apply([inputs, output])
        })
    }

    dense(x, kernel, bias) {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
        const m = tf.matMul(x, k)
        if (bias) {
            return tf.add(m, bias.read())
        } else {
            return m
        }
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
            this.bias.read()
        ]
    }

    setWeights(weights) {
        this.kernel.write(weights[0])
        this.outputKernel.write(weights[1])
        this.recurrentKernel.write(weights[2])
        this.bias.write(weights[3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            epsilon: this.epsilon,
            k: this.k
        }
    }
}

class Interrogator extends LayerBase {
    constructor(config) {
        super({ name: `int-${randomString()}`, ...config })
        this.units = config?.units || 64
        this.maxDecisions = config?.maxDecisions || 3
        this.kernelSize = config?.kernelSize || 3
        this.dilation = config?.dilation || 1
        this.gamma = config?.gamma || 2
    }

    build(inputShape) {
        this.router = this.addWeight(
            'routingMatrix',
            [inputShape[inputShape.length - 1], this.units],
            'float32',
            tf.initializers.leCunNormal()
        )

        this.lens = customLayers.conv1d({
            filters: this.units,
            kernelSize: this.kernelSize,
            kernelInitializer: 'heNormal',
            dilationRate: this.dilation,
            padding: 'same',
            activation: 'mish',
            useBias: true
        })

        this.attention = customLayers.EfficientChannelAttention({
            gamma: this.gamma
        })

        this.gate = this.addWeight(
            'gateVector',
            [this.units],
            'float32',
            tf.initializers.glorotUniform()
        )

        this.focus = this.addWeight(
            'focusVector',
            [this.units],
            'float32',
            tf.initializers.glorotNormal()
        )

        this.alpha = tf.variable(tf.scalar(0.5))

        this.residual = customLayers.ResidualConnection()

        this.mask = tf.linalg
            .bandPart(tf.ones([inputShape[1], inputShape[2]]), 0, -1)
            .expandDims(0)
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const targets = tf.variable(
                tf.randomUniform(inputs.shape, -0.0001, 0.0001)
            )

            const masked = inputs.mul(this.mask.tile([inputs.shape[0], 1, 1]))

            let outputs = masked

            for (let i = 0; i < this.maxDecisions; i++) {
                const focus = this.lens.apply(outputs)

                const intents = this.attention.apply(focus)

                const routes = intents.matMul(this.router.read()).selu()

                this.alpha.assign(
                    tf.sigmoid(routes.mul(this.focus.read())).mean()
                )

                const transitions = tf
                    .prelu(routes.mul(this.gate.read()), this.alpha)
                    .transpose([0, 2, 1])

                const scores = tf.matMul(
                    outputs.transpose([0, 2, 1]),
                    transitions,
                    true
                )

                const weights = tf.softmax(scores, -1)

                const direction = tf
                    .matMul(transitions, weights)
                    .transpose([0, 2, 1])

                targets.assign(targets.add(direction))

                outputs = targets
            }

            targets.dispose()

            return this.residual.apply([inputs, outputs])
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            maxDecisions: this.maxDecisions,
            kernelSize: this.kernelSize,
            dilation: this.dilation,
            gamma: this.gamma
        }
    }
}

class TemporalPooling extends LayerBase {
    constructor(config) {
        super({ name: `con-${randomString()}`, ...config })
        this.minKernelSize = config.minKernelSize || 2
        this.maxKernelSize = config.maxKernelSize || 5
        this.poolingFunction = config.poolingFunction || 'max'
        this.padding = config.padding || 'valid'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.kernelSizes = []

        for (
            let size = this.minKernelSize;
            size <= this.maxKernelSize;
            size++
        ) {
            const kernelSize = size
            this.kernelSizes.push(kernelSize)
        }

        this.numKernels = this.kernelSizes.length

        this.kernels = []
        for (let i = 0; i < this.numKernels; i++) {
            const kernel = this.addWeight(
                `kernel_${i}`,
                [this.kernelSizes[i], inputDim, inputDim],
                'float32',
                tf.initializers.glorotUniform()
            )
            this.kernels.push(kernel)
        }
    }

    call(inputs) {
        return tf.tidy(() => {
            const convResults = []

            for (let i = 0; i < this.numKernels; i++) {
                const convResult = tf.tidy(() => {
                    const kernel = this.kernels[i].read()
                    const conv = tf.conv1d(inputs, kernel, 1, this.padding)
                    return conv
                })
                convResults.push(convResult)
            }

            const pooledResults = []
            for (let i = 0; i < this.numKernels; i++) {
                const pooled = tf.tidy(() => {
                    if (this.poolingFunction === 'max') {
                        return tf.max(convResults[i], 1, true)
                    } else if (this.poolingFunction === 'avg') {
                        return tf.mean(convResults[i], 1, true)
                    }
                })
                pooledResults.push(pooled)
            }

            const concatenated = tf.concat(pooledResults, -1)
            const output = tf.reshape(concatenated, inputs.shape)

            return output
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            minKernelSize: this.minKernelSize,
            maxKernelSize: this.maxKernelSize,
            poolingFunction: this.poolingFunction,
            padding: this.padding
        }
    }
}

class PseudoQuantumState extends LayerBase {
    constructor(config) {
        super({ name: `qua-${randomString()}`, ...config })
        this.units = config.units
        this.qubits = config.qubits || 4
    }

    build(inputShape) {
        this.quantumWeights = this.addWeight(
            'quantumWeights',
            [this.units, this.qubits],
            'float32',
            tf.initializers.heNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.entanglementMatrix = this.addWeight(
            'entanglementMatrix',
            [this.qubits, this.qubits],
            'float32',
            tf.initializers.heNormal()
        )
        this.classicalWeights = this.addWeight(
            'classicalWeights',
            [this.qubits, this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.seedVector = this.addWeight(
            'seedVector',
            [this.units],
            'float32',
            tf.initializers.randomNormal({ mean: 0, stddev: 1 })
        )
        this.scalingFactor = this.addWeight(
            'scalingFactor',
            [],
            'float32',
            tf.initializers.constant({ value: 1 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const batchSize = inputs.shape[0]
            const sequenceLength = inputs.shape[1]
            const inputDim = inputs.shape[2]

            // Generate seed value based on inputs and seed vector
            const reshaped = tf.reshape(inputs, [-1, inputDim])
            const seedWeights = tf.dot(reshaped, this.seedVector.read())
            const seed = tf.mean(seedWeights).dataSync()[0]

            // Prepare quantum states from inputs
            const initialStates = tf
                .reshape(
                    tf.matMul(
                        tf.reshape(inputs, [-1, inputDim]),
                        this.quantumWeights.read()
                    ),
                    [batchSize, sequenceLength, this.qubits]
                )
                .tanh()

            // Apply quantum entanglement
            const entangledStates = tf
                .reshape(
                    tf.matMul(
                        tf.reshape(initialStates, [-1, this.qubits]),
                        this.entanglementMatrix.read()
                    ),
                    [batchSize, sequenceLength, this.qubits]
                )
                .mul(this.scalingFactor.read())
                .softmax()

            // Perform measurement and collapse using Gumbel-Softmax trick
            const temperature = 1.0
            const state = tf.randomUniform(
                entangledStates.shape,
                0,
                1,
                'float32',
                seed
            )
            const noisyLogits = tf.add(tf.log(tf.abs(entangledStates)), state)
            const samples = tf.softmax(tf.div(noisyLogits, temperature))
            const measurements = tf.argMax(samples, -1)

            // Classical post-processing
            const outcomes = tf
                .oneHot(measurements, this.qubits)
                .reshape([-1, this.qubits])

            const outputs = tf.matMul(outcomes, this.classicalWeights.read())

            return tf.reshape(outputs, [batchSize, sequenceLength, this.units])
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            qubits: this.qubits
        }
    }
}

class Collideoscope extends LayerBase {
    constructor(config) {
        super({ name: `att-${randomString()}`, ...config })
        this.units = config.units
        this.blockSize = config.blockSize
    }

    build(inputShape) {
        console.log('Input shape:', inputShape)
        console.log('Units:', this.units)
        console.log('Block size:', this.blockSize)

        this.queryWeights = this.addWeight(
            'query',
            [this.units, this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.keyWeights = this.addWeight(
            'key',
            [this.units, this.units],
            'float32',
            tf.initializers.glorotUniform()
        )

        console.log('Query weights shape:', this.queryWeights.shape)
        console.log('Key weights shape:', this.keyWeights.shape)
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.blockSize, this.units]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            console.log('Input tensor shape:', inputs.shape)

            const transformedQuery = tf.matMul(inputs, this.queryWeights.read())
            const transformedKey = tf.matMul(inputs, this.keyWeights.read())

            console.log('Transformed query shape:', transformedQuery.shape)
            console.log('Transformed key shape:', transformedKey.shape)

            const scores = tf.matMul(
                transformedQuery,
                transformedKey,
                false,
                true
            )
            const attentionWeights = tf.softmax(scores, -1)

            console.log('Attention weights shape:', attentionWeights.shape)

            const outputTensor = tf.matMul(attentionWeights, inputs)

            console.log('Output tensor shape:', outputTensor.shape)

            return outputTensor
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            blockSize: this.blockSize
        }
    }
}

// class Collideoscope extends LayerBase {
//     constructor(config) {
//         super({ name: `ssm-${randomString()}`, ...config })
//         this.units = config.units || 64
//         this.innerDim = config.innerDim || 256
//         this.returnSequences = config.returnSequences || false
//     }

//     build(inputShape) {
//         const inputDim = inputShape[2]
//         this.kernel = this.addWeight(
//             'kernel',
//             [inputDim, this.innerDim],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.recurrentKernel = this.addWeight(
//             'recurrentKernel',
//             [this.units, this.innerDim],
//             'float32',
//             tf.initializers.orthogonal({ gain: 1 })
//         )
//         this.outputKernel = this.addWeight(
//             'outputKernel',
//             [this.innerDim, this.units],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//         this.bias = this.addWeight(
//             'bias',
//             [this.innerDim],
//             'float32',
//             tf.initializers.zeros()
//         )

//         this.residual = new ResidualConnection()
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             const [batchSize, sequenceLength, inputDim] = inputs.shape

//             let state = tf.zeros([batchSize, this.units])
//             const outputs = []

//             const kernel = this.kernel.read()
//             const recurrentKernel = this.recurrentKernel.read()
//             const outputKernel = this.outputKernel.read()
//             const bias = this.bias.read()

//             for (let t = 0; t < sequenceLength; t++) {
//                 const input = inputs
//                     .slice([0, t, 0], [batchSize, 1, inputDim])
//                     .reshape([batchSize, inputDim])
//                 const innerState = tf
//                     .add(
//                         tf.matMul(input, kernel),
//                         tf.matMul(state, recurrentKernel)
//                     )
//                     .add(bias)
//                     .tanh()
//                 const newState = tf.matMul(innerState, outputKernel)
//                 outputs.push(newState)
//                 state = newState
//             }

//             let output = this.returnSequences
//                 ? tf.stack(outputs, 1)
//                 : outputs[outputs.length - 1]

//             return this.residual.apply([inputs, output])
//         })
//     }

//     computeOutputShape(inputShape) {
//         const outputShape = this.returnSequences
//             ? [inputShape[0], inputShape[1], this.units]
//             : [inputShape[0], this.units]
//         return outputShape
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             units: this.units,
//             innerDim: this.innerDim,
//             returnSequences: this.returnSequences
//         }
//     }
// }

// class Collideoscope extends LayerBase {
//     constructor(config) {
//         super({ name: `qs-${randomString()}`, ...config })
//         this.units = config.units || 64
//         this.qubits = config.qubits || 4
//         this.iterations = config.iterations || 1
//     }

//     build(inputShape) {
//         this.entryWeights = this.addWeight(
//             `entryWeights`,
//             [this.units, this.qubits],
//             'float32',
//             tf.initializers.glorotUniform()
//         )

//         this.transitions = this.addWeight(
//             `transitions`,
//             [this.units + this.qubits, this.qubits * 4],
//             'float32',
//             tf.initializers.glorotUniform()
//         )

//         this.bias = this.addWeight(
//             `bias`,
//             [this.qubits * 4],
//             'float32',
//             tf.initializers.zeros()
//         )

//         this.propulsion = []
//         this.collapse = []
//         this.expansionFactor = []
//         this.attractionFactor = []
//         this.expansionBias = []
//         this.collapseBias = []
//         this.upperMagnitude = []
//         this.lowerMagnitude = []
//         this.alpha = []
//         this.prism = []

//         for (let i = 0; i < this.iterations; i++) {
//             this.propulsion.push(
//                 this.addWeight(
//                     `propulsion${i}`,
//                     [this.qubits, this.qubits],
//                     'float32',
//                     tf.initializers.glorotUniform()
//                 )
//             )
//             this.collapse.push(
//                 this.addWeight(
//                     `collapse${i}`,
//                     [this.qubits, this.qubits],
//                     'float32',
//                     tf.initializers.orthogonal({ gain: 1 })
//                 )
//             )
//             this.expansionFactor.push(
//                 this.addWeight(
//                     `expansionFactor${i}`,
//                     [],
//                     'float32',
//                     tf.initializers.ones()
//                 )
//             )
//             this.expansionBias.push(
//                 this.addWeight(
//                     `expansionBias${i}`,
//                     [this.qubits],
//                     'float32',
//                     tf.initializers.zeros()
//                 )
//             )
//             this.attractionFactor.push(
//                 this.addWeight(
//                     `attractionFactor${i}`,
//                     [],
//                     'float32',
//                     tf.initializers.ones()
//                 )
//             )
//             this.collapseBias.push(
//                 this.addWeight(
//                     `collapseBias${i}`,
//                     [this.qubits],
//                     'float32',
//                     tf.initializers.zeros()
//                 )
//             )
//             this.upperMagnitude.push(
//                 this.addWeight(
//                     `upperMagnitude${i}`,
//                     [],
//                     'float32',
//                     tf.initializers.ones()
//                 )
//             )
//             this.lowerMagnitude.push(
//                 this.addWeight(
//                     `lowerMagnitude${i}`,
//                     [],
//                     'float32',
//                     tf.initializers.zeros()
//                 )
//             )
//             this.alpha.push(
//                 this.addWeight(
//                     `alpha${i}`,
//                     [],
//                     'float32',
//                     tf.initializers.constant({ value: 0.22 })
//                 )
//             )
//             this.prism.push(
//                 this.addWeight(
//                     `prism${i}`,
//                     [2],
//                     'float32',
//                     tf.initializers.glorotUniform()
//                 )
//             )
//         }

//         this.exitWeights = this.addWeight(
//             `exitWeights`,
//             [this.qubits, this.units],
//             'float32',
//             tf.initializers.glorotUniform()
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             this.mask = tf.linalg
//                 .bandPart(tf.ones([inputs.shape[1], inputs.shape[2]]), 0, -1)
//                 .expandDims(0)

//             const masked = inputs.mul(this.mask.tile([inputs.shape[0], 1, 1]))

//             // Transform the input into quantum states
//             const initialStates = tf
//                 .matMul(
//                     tf.reshape(masked, [-1, inputs.shape[2]]),
//                     this.entryWeights.read()
//                 )
//                 .reshape([-1, this.qubits])

//             let evolvingStates = initialStates

//             let c = tf.zeros([evolvingStates.shape[0], this.qubits])
//             let h = tf.zeros([evolvingStates.shape[0], this.qubits])

//             let v = tf.randomUniform([1], 0, 1)

//             for (let i = 0; i < this.iterations; i++) {
//                 // Simulate quantum expansion (non-linear activation)
//                 evolvingStates = tf
//                     .matMul(evolvingStates, this.propulsion[i].read())
//                     .mul(
//                         customOps.subliminalSpace(
//                             this.lowerMagnitude[i].read(),
//                             this.upperMagnitude[i].read(),
//                             evolvingStates.shape[1]
//                         )
//                     )
//                     .mul(this.expansionFactor[i].read())
//                     .add(this.expansionBias[i].read())
//                     .prelu(this.alpha[i].read())

//                 // Apply a transition function
//                 const p = this.prism[i].read()
//                 v = tf.concat([p, v])
//                 ;[c, h] = tf.basicLSTMCell(
//                     tf.outerProduct(p, v).mean(),
//                     this.transitions.read(),
//                     this.bias.read(),
//                     tf.concat([evolvingStates, h], 1),
//                     c,
//                     h
//                 )

//                 // Simulate quantum collapse (linear transformation)
//                 evolvingStates = tf
//                     .matMul(h, this.collapse[i].read())
//                     .sub(this.collapseBias[i].read())
//                     .mul(this.attractionFactor[i].read().neg()) // division

//                 // Carry information via residual connections
//                 evolvingStates = tf.add(initialStates, evolvingStates)
//             }

//             // Classical post-processing
//             const output = tf.reshape(
//                 tf.matMul(evolvingStates, this.exitWeights.read()),
//                 inputs.shape
//             )

//             return output
//         })
//     }

//     computeOutputShape(inputShape) {
//         return inputShape
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             units: this.units,
//             qubits: this.qubits,
//             iterations: this.iterations
//         }
//     }
// }

class QuantumStateMachine extends LayerBase {
    constructor(config) {
        super({ name: `qsm-${randomString()}`, ...config })
        this.units = config.units || 64
        this.qubits = config.qubits || 4
        this.iterations = config.iterations || 1
    }

    build(inputShape) {
        this.entryWeights = this.addWeight(
            `entryWeights`,
            [this.units, this.qubits],
            'float32',
            tf.initializers.glorotUniform()
        )

        this.quantumWeights = []
        this.quantumGates = []
        this.expansionFactor = []
        this.collapseFactor = []

        for (let i = 0; i < this.iterations; i++) {
            this.quantumGates.push(
                this.addWeight(
                    `quantumGates${i}`,
                    [this.qubits, this.qubits],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
            this.quantumWeights.push(
                this.addWeight(
                    `quantumWeights${i}`,
                    [this.qubits, this.qubits],
                    'float32',
                    tf.initializers.orthogonal({ gain: 1 })
                )
            )
            this.expansionFactor.push(
                this.addWeight(
                    `expansionFactor${i}`,
                    [],
                    'float32',
                    tf.initializers.ones()
                )
            )
            this.collapseFactor.push(
                this.addWeight(
                    `collapseFactor${i}`,
                    [],
                    'float32',
                    tf.initializers.ones()
                )
            )
        }

        this.classicalWeights = this.addWeight(
            `classicalWeights`,
            [this.qubits, this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Transform the input into quantum states
            let evolvingStates = tf
                .matMul(
                    tf.reshape(inputs, [-1, inputs.shape[2]]),
                    this.entryWeights.read()
                )
                .reshape([-1, this.qubits])

            for (let i = 0; i < this.iterations; i++) {
                // Apply quantum gates (non-linear activation)
                evolvingStates = tf
                    .matMul(evolvingStates, this.quantumGates[i].read())
                    .mul(this.expansionFactor[i].read())
                    .selu()

                // Apply quantum weights (linear transformation)
                evolvingStates = tf
                    .matMul(evolvingStates, this.quantumWeights[i].read())
                    .mul(this.collapseFactor[i].read().neg()) // division
            }

            // Classical post-processing
            return tf.reshape(
                tf.matMul(evolvingStates, this.classicalWeights.read()),
                inputs.shape
            )
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            qubits: this.qubits,
            iterations: this.iterations
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

class CollapseOneHot extends LayerBase {
    constructor(config) {
        super(config)
    }

    computeOutputShape(inputShape) {
        return inputShape.slice(0, -1)
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            return tf.cast(inputs, 'int32')
        })
    }
}

class ToOneHot extends LayerBase {
    constructor(config) {
        super(config)
        this.depth = config.depth
    }

    computeOutputShape(inputShape) {
        return [...inputShape.slice(0, -1), this.depth]
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            return tf.oneHot(inputs.cast('int32'), this.depth)
        })
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, { depth: this.depth })
        return config
    }
}

class DeterministicEmbedding extends LayerBase {
    constructor(config) {
        super({ name: `emb-${randomString()}`, ...config })
        this.outputDim = config.outputDim
    }

    computeOutputShape(inputShape) {
        return [...inputShape, this.outputDim]
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const tokenIds = inputs.cast('int32')
            const positions = tf
                .range(0, inputs.shape[1])
                .expandDims(0)
                .cast('int32')

            const tokenEncodings = tf
                .oneHot(tokenIds, this.outputDim)
                .cast('float32')
            const positionEncodings = tf
                .oneHot(positions, this.outputDim)
                .cast('float32')

            const encodings = tokenEncodings.add(positionEncodings)
            const normalizedEncodings = encodings.div(
                tf.sqrt(tf.scalar(this.outputDim))
            )

            return normalizedEncodings
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim
        }
    }
}

class PerformerAttention extends LayerBase {
    constructor(config) {
        super({ name: `perf-${randomString()}`, ...config })
        this.heads = config.heads
        this.units = config.units
        this.projectionDim = config.projectionDim
    }

    build(inputShape) {
        const [batchSize, sequenceLength, inputDim] = inputShape
        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputDim, this.heads * this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputDim, this.heads * this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputDim, this.heads * this.projectionDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.heads * this.projectionDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.attentionWeights = []
        this.attentionOutputs = []
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, sequenceLength, inputDim] = inputs.shape

            const queries = tf.matMul(inputs, this.queryKernel.read())
            const keys = tf.matMul(inputs, this.keyKernel.read())
            const values = tf.matMul(inputs, this.valueKernel.read())

            const queriesReshaped = tf.reshape(queries, [
                batchSize,
                sequenceLength,
                this.heads,
                this.units
            ])
            const keysReshaped = tf.reshape(keys, [
                batchSize,
                sequenceLength,
                this.heads,
                this.units
            ])
            const valuesReshaped = tf.reshape(values, [
                batchSize,
                sequenceLength,
                this.heads,
                this.projectionDim
            ])

            const queriesPerHead = queriesReshaped.transpose([0, 2, 1, 3])
            const keysPerHead = keysReshaped.transpose([0, 2, 1, 3])
            const valuesPerHead = valuesReshaped.transpose([0, 2, 1, 3])

            console.log('queriesPerHead shape:', queriesPerHead.shape)
            console.log('keysPerHead shape:', keysPerHead.shape)
            console.log('valuesPerHead shape:', valuesPerHead.shape)

            const attentionScores = this.favorAttention(
                queriesPerHead,
                keysPerHead
            )
            console.log('attentionScores shape:', attentionScores.shape)

            const attentionWeights = this.applyMask(attentionScores)
            console.log('attentionWeights shape:', attentionWeights.shape)
            console.log('valuesPerHead shape:', valuesPerHead.shape)

            const attentionOutputs = tf.matMul(
                attentionWeights.transpose([0, 1, 3, 2]),
                valuesPerHead
            )
            console.log('attentionOutputs shape:', attentionOutputs.shape)

            const outputTransposed = attentionOutputs.transpose([0, 2, 1, 3])
            const concatOutput = tf.reshape(outputTransposed, [
                batchSize,
                sequenceLength,
                this.heads * this.projectionDim
            ])
            const output = tf.matMul(concatOutput, this.outputKernel.read())

            return output
        })
    }

    favorAttention(queries, keys) {
        const [batchSize, numHeads, sequenceLength, keyDim] = queries.shape
        const numRandomFeatures = Math.floor(sequenceLength / 2) + 1

        const randomMatrices = []
        for (let i = 0; i < numHeads; i++) {
            const randomMatrix = tf.randomNormal([keyDim, numRandomFeatures])
            randomMatrices.push(randomMatrix)
        }

        const randomFeatures = []
        for (let i = 0; i < numHeads; i++) {
            const queryRandomFeatures = tf.matMul(
                queries.gather([i], 1).squeeze([1]),
                randomMatrices[i]
            )
            const keyRandomFeatures = tf.matMul(
                keys.gather([i], 1).squeeze([1]),
                randomMatrices[i]
            )

            const queryRandomFeaturesExp = queryRandomFeatures.exp()
            const keyRandomFeaturesExp = keyRandomFeatures.exp()

            randomFeatures.push({
                queries: queryRandomFeaturesExp,
                keys: keyRandomFeaturesExp
            })
        }

        const attentionScores = []
        for (let i = 0; i < numHeads; i++) {
            const queryRandomFeatures = randomFeatures[i].queries
            const keyRandomFeatures = randomFeatures[i].keys

            const dotProduct = tf.matMul(
                queryRandomFeatures,
                keyRandomFeatures,
                false,
                true
            )
            const scaledDotProduct = dotProduct.div(tf.sqrt(tf.scalar(keyDim)))

            attentionScores.push(scaledDotProduct)
        }

        const attentionScoresStacked = tf.stack(attentionScores, 1)
        return attentionScoresStacked
    }

    applyMask(attentionScores) {
        const attentionWeights = tf.softmax(attentionScores, -1)
        return attentionWeights
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            heads: this.heads,
            units: this.units,
            projectionDim: this.projectionDim
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

class DimensionExpansion extends LayerBase {
    constructor(config) {
        super({ name: `exp-${randomString()}`, ...config })
        this.units = config.units
        this.method = config.method || ['fluid', 'tiled'][0]
    }

    build(inputShape) {
        this.upperAlpha = 0.88888888
        this.lowerAlpha = 0.00000008
        this.current = tf.variable(tf.scalar(1.0))
        this.valve = tf.variable(tf.scalar(this.upperAlpha))
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const iterations = Math.floor(this.units / inputs.shape[2])
            const repeatedInputs = this[this.method](inputs, iterations)
            const paddedOutputs = tf.pad(repeatedInputs, [
                [0, 0],
                [0, 0],
                [0, this.units - iterations * inputs.shape[2]]
            ])
            return paddedOutputs
        })
    }

    fluid(inputs, iterations) {
        const tiles = []
        const weights = inputs.softmax()

        for (let i = 0; i < iterations; i++) {
            const alphaCurrent = this.current
                .sigmoid()
                .mul(this.upperAlpha - this.lowerAlpha)
                .add(this.lowerAlpha)
            const alphaWeight = this.valve
                .mul(alphaCurrent)
                .clipByValue(this.lowerAlpha, this.upperAlpha)

            const alpha = alphaWeight.dataSync()[0]

            if (Math.random() < 0.001) console.log(alpha)

            let tile = inputs
            if (i !== 0) tile = tf.leakyRelu(inputs.mul(i), alpha)
            tiles.push(tile)

            const weight = tile.mul(weights).sum().div(weights.sum())

            this.valve.assign(weight)
            this.current.assign(
                this.current.add(
                    alphaWeight
                        .sub(this.lowerAlpha)
                        .div(this.upperAlpha - this.lowerAlpha)
                )
            )
        }
        return tf.concat(tiles, 2)
    }

    tiled(inputs, iterations) {
        return tf.tile(inputs, [1, 1, iterations])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units
        }
    }
}

class DimensionContraction extends LayerBase {
    constructor(config) {
        super({ name: `con-${randomString()}`, ...config })
        this.units = config.units
        this.activation = config.activation || 'tanh'
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const inputShape = inputs.shape
            const dimensions = inputShape[2]
            const chunkSize = Math.floor(dimensions / this.units)
            const paddedDimensions = chunkSize * this.units
            const paddedInputs = inputs.pad([
                [0, 0],
                [0, 0],
                [0, paddedDimensions - dimensions]
            ])
            const reshapedInputs = paddedInputs.reshape([
                inputShape[0],
                inputShape[1],
                this.units,
                chunkSize
            ])

            const chunks = []
            for (let i = 0; i < this.units; i++) {
                const chunk = reshapedInputs
                    .slice([0, 0, i, 0], [-1, -1, 1, -1])
                    .squeeze([2])
                const activated = tf[this.activation](chunk)
                const reduced = activated.sum([2], true)
                chunks.push(reduced)
            }
            return tf.concat(chunks, 2)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            activation: this.activation
        }
    }
}

class DepthwiseSeparableConvolution extends LayerBase {
    constructor(config) {
        super({ name: `con-${randomString()}`, ...config })
        this.units = config?.units || 256
        // this.dropout = config?.dropout || 0
        // this.epsilon = config?.epsilon || 1e-5
        this.activation = config?.activation || 'relu'
        this.supportsMasking = true
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        // Initialize depthwise convolution layer
        this.depthwiseConv = tf.layers.depthwiseConv2d({
            kernelSize: [1, 1],
            depthMultiplier: 1,
            activation: this.activation,
            useBias: false,
            dataFormat: 'channelsLast'
        })

        // Initialize pointwise convolution layer
        this.pointwiseConv = tf.layers.conv2d({
            filters: this.units,
            kernelSize: [1, 1],
            activation: 'linear',
            useBias: false,
            dataFormat: 'channelsLast'
        })

        // Initialize layer normalization
        // this.layernorm = tf.layers.layerNormalization({
        //     epsilon: this.epsilon
        // })

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand dimensions to match the expected input shape of depthwiseConv2d
            const expandedInputs = inputs.expandDims(2)

            // Apply depthwise convolution
            let outputs = this.depthwiseConv.apply(expandedInputs)

            // Apply pointwise convolution
            outputs = this.pointwiseConv.apply(outputs)

            // Squeeze dimensions to match the original input shape
            outputs = outputs.squeeze([2])

            // If training, apply residual dropout
            // outputs = kwargs['training']
            //     ? tf.dropout(outputs, this.dropout)
            //     : outputs

            // Apply layer norm
            // outputs = this.layernorm.apply(outputs)

            // Apply skip connection
            return this.residual.apply([inputs, outputs])
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            units: this.units,
            activation: this.activation
            // dropout: this.dropout
        }
    }
}

class VectorLayerWithMixing extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units
        this.mixingSize = config.mixingSize
    }

    build(inputShape) {
        this.kernel = this.addWeight(
            'kernel',
            [inputShape[inputShape.length - 1]],
            'float32',
            tf.initializers.randomNormal()
        )
        this.bias = this.addWeight(
            'bias',
            [this.units],
            'float32',
            tf.initializers.zeros()
        )
    }

    call(inputs) {
        // Apply token mixing to the input
        const mixedInput = this.applyMixing(inputs)

        // Perform element-wise multiplication with the weight vector
        const dotProduct = tf.mul(mixedInput, this.kernel)

        // Apply token mixing to the intermediate state
        const mixedIntermediate = this.applyMixing(dotProduct)

        // Add the bias vector
        const output = tf.add(mixedIntermediate, this.bias)

        return output
    }

    applyMixing(tensor) {
        const shape = tensor.shape
        const [batchSize, sequenceLength, depth] = shape

        // Reshape the tensor to a matrix
        const matrix = tf.reshape(tensor, [batchSize * sequenceLength, depth])

        // Split the matrix into chunks
        const chunks = tf.split(matrix, this.mixingSize, 1)

        // Shuffle the chunks along the depth dimension
        const shuffledChunks = tf.stack(tf.shuffle(chunks))

        // Reshape the shuffled chunks back to the original tensor shape
        const reshapedTensor = tf.reshape(shuffledChunks, shape)

        return reshapedTensor
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            units: this.units,
            mixingSize: this.mixingSize
        })
        return config
    }
}

class InstanceNormalization extends LayerBase {
    constructor(config) {
        super(config)
        this.epsilon = 1e-5
    }

    build(inputShape) {
        const lastDim = inputShape[inputShape.length - 1]
        this.gamma = this.addWeight(
            'gamma',
            [lastDim],
            'float32',
            tf.initializers.ones()
        )
        this.beta = this.addWeight(
            'beta',
            [lastDim],
            'float32',
            tf.initializers.zeros()
        )
    }

    call(inputs, kwargs) {
        const mean = tf.mean(inputs, [1, 2], true)
        const variance = tf.mean(tf.square(tf.sub(inputs, mean)), [1, 2], true)
        const std = tf.sqrt(tf.add(variance, this.epsilon))
        const outputs = tf.div(tf.sub(inputs, mean), std)
        const gammaExpanded = tf.expandDims(tf.expandDims(this.gamma, 0), 0)
        const betaExpanded = tf.expandDims(tf.expandDims(this.beta, 0), 0)
        return tf.add(tf.mul(outputs, gammaExpanded), betaExpanded)
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, { epsilon: this.epsilon })
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

// https://arxiv.org/abs/1610.06258
class FastMemory extends LayerBase {
    constructor(config) {
        super({ name: `mem-${randomString()}`, ...config })
        this.units = config.units || 64
        this.forgetfulness = config.forgetfulness || 0.9
        this.static = config.static || false
    }

    build(inputShape) {
        this.inputDim = inputShape[inputShape.length - 1]
        this.fastWeights = tf.variable(
            tf.randomNormal([this.inputDim, this.units], 0, 0.1),
            false
        )
        this.currentState = tf.variable(
            tf.ones([this.inputDim, this.units]),
            false
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, sequenceLength, inputDim] = inputs.shape

            // Pad sequences with zeros to match the maximum sequence length
            if (sequenceLength < this.inputDim) {
                const padAmount = this.inputDim - sequenceLength
                inputs = tf.pad(inputs, [
                    [0, 0],
                    [0, padAmount],
                    [0, 0]
                ])
            }

            const newStates = []

            for (let i = 0; i < batchSize; i++) {
                const currentExample = inputs.slice(
                    [i, 0, 0],
                    [1, this.inputDim, this.inputDim]
                )

                let newWeights = tf.matMul(currentExample, this.fastWeights)
                newWeights = newWeights.div(newWeights.norm())

                let newState = tf.add(
                    tf.mul(this.forgetfulness, this.currentState),
                    tf.mul(1 - this.forgetfulness, newWeights)
                )
                newState = newState.div(newState.norm())

                if (this.static) {
                    newState = newState.add(
                        tf.randomNormal(
                            newState.shape,
                            -this.static,
                            this.static
                        )
                    )
                }

                newStates.push(newState)

                this.fastWeights.assign(newWeights.squeeze())
                this.currentState.assign(newState.squeeze())
            }

            const outputs = tf.concat(newStates, 0)

            return this.residual.apply([inputs, outputs])
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            forgetfulness: this.forgetfulness,
            static: this.static
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

            const Q = this.applyDense(inputs, this.toQ).reshape([
                B,
                T,
                this.hiddenDim
            ])
            const K = this.applyDense(inputs, this.toK).reshape([
                B,
                T,
                this.hiddenDim
            ])
            const V = this.applyDense(inputs, this.toV).reshape([
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
                this.project
            )

            return this.residual.apply([inputs, outputs])
        })
    }

    applyDense(x, kernel) {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
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

// class PrincipalComponentAnalysis extends LayerBase {
//     constructor(config) {
//         super({ name: `pca-${randomString()}`, ...config })
//         this.outputDim = config.outputDim
//         this.epsilon = config.epsilon || 1e-7
//         this.maxIterations = config.maxIterations || 1000
//         this.centered = false
//         this.mean = null
//         this.components = null
//         this.explainedVariance = null
//         this.debugMode = config.debugMode || false
//     }

//     build(inputShape) {
//         const inputDim = inputShape[inputShape.length - 1]
//         this.inputDim = inputDim
//         if (this.debugMode)
//             console.log(`PCA: Input dimension is ${this.inputDim}`)
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs
//             if (this.debugMode) console.log(`PCA: Input shape: ${inputs.shape}`)

//             if (!this.centered) {
//                 if (this.debugMode) console.log('PCA: Fitting data...')
//                 this.fit(inputs)
//             }

//             // Center the data
//             const centered = tf.sub(inputs, this.mean)
//             if (this.debugMode)
//                 console.log(`PCA: Centered data shape: ${centered.shape}`)

//             // Project data onto principal components
//             const flattenedCentered = tf.reshape(centered, [-1, this.inputDim])
//             if (this.debugMode)
//                 console.log(
//                     `PCA: Flattened centered data shape: ${flattenedCentered.shape}`
//                 )

//             const flattenedResult = tf.matMul(
//                 flattenedCentered,
//                 this.components
//             )
//             if (this.debugMode)
//                 console.log(
//                     `PCA: Flattened result shape: ${flattenedResult.shape}`
//                 )

//             const result = tf.reshape(flattenedResult, [
//                 ...inputs.shape.slice(0, -1),
//                 this.outputDim
//             ])
//             if (this.debugMode)
//                 console.log(`PCA: Final result shape: ${result.shape}`)

//             return result
//         })
//     }

//     fit(X) {
//         tf.tidy(() => {
//             if (this.debugMode)
//                 console.log(`PCA: Fitting data with shape ${X.shape}`)

//             // Center the data
//             this.mean = tf.mean(X, [0, 1], true)
//             if (this.debugMode)
//                 console.log(`PCA: Computed mean with shape ${this.mean.shape}`)

//             const centered = tf.sub(X, this.mean)
//             if (this.debugMode)
//                 console.log(`PCA: Centered data shape: ${centered.shape}`)

//             // Compute covariance matrix
//             const flattenedCentered = tf.reshape(centered, [-1, this.inputDim])
//             if (this.debugMode)
//                 console.log(
//                     `PCA: Flattened centered data shape: ${flattenedCentered.shape}`
//                 )

//             const scaleFactor = 1 / (flattenedCentered.shape[0] - 1)
//             const cov = tf.tidy(() => {
//                 const covMatrix = tf.matMul(
//                     flattenedCentered,
//                     flattenedCentered,
//                     true,
//                     false
//                 )
//                 return tf.mul(covMatrix, scaleFactor)
//             })
//             if (this.debugMode)
//                 console.log(`PCA: Covariance matrix shape: ${cov.shape}`)

//             // Compute principal components using power iteration
//             try {
//                 const { components, eigenvalues } = this.powerIteration(
//                     cov,
//                     this.outputDim
//                 )
//                 this.components = components
//                 this.explainedVariance = eigenvalues

//                 if (this.debugMode) {
//                     console.log(
//                         `PCA: Components shape: ${this.components.shape}`
//                     )
//                     console.log(
//                         `PCA: Explained variance shape: ${this.explainedVariance.shape}`
//                     )
//                 }

//                 this.centered = true
//             } catch (error) {
//                 console.error('Error during power iteration:', error)
//                 throw new Error(
//                     'PCA computation failed. The input data might be ill-conditioned or contain invalid values.'
//                 )
//             }
//         })
//     }

//     powerIteration(matrix, numComponents) {
//         return tf.tidy(() => {
//             const [n, m] = matrix.shape
//             let components = tf.randomNormal([n, numComponents])
//             let eigenvalues = tf.zeros([numComponents])

//             for (let i = 0; i < numComponents; i++) {
//                 let eigenvector = components.slice([0, i], [-1, 1])
//                 let eigenvalue = tf.scalar(0)

//                 for (let iter = 0; iter < this.maxIterations; iter++) {
//                     // Power iteration
//                     eigenvector = tf.matMul(matrix, eigenvector)
//                     eigenvalue = tf.norm(eigenvector)
//                     eigenvector = eigenvector.div(eigenvalue)

//                     // Check for convergence
//                     if (
//                         iter > 0 &&
//                         Math.abs(
//                             eigenvalue.sub(tf.scalar(eigenvalue)).dataSync()[0]
//                         ) < this.epsilon
//                     ) {
//                         break
//                     }
//                 }

//                 // Deflate the matrix
//                 const outer = tf.matMul(eigenvector, eigenvector, false, true)
//                 matrix = matrix.sub(outer.mul(eigenvalue))

//                 // Store the component and eigenvalue
//                 components.slice([0, i], [-1, 1]).assign(eigenvector)
//                 eigenvalues.slice([i], [1]).assign(eigenvalue)
//             }

//             return { components, eigenvalues }
//         })
//     }

//     computeOutputShape(inputShape) {
//         return [...inputShape.slice(0, -1), this.outputDim]
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             outputDim: this.outputDim,
//             epsilon: this.epsilon,
//             maxIterations: this.maxIterations,
//             debugMode: this.debugMode
//         }
//     }
// }

// class PrincipalComponentAnalysis extends LayerBase {
//     constructor(config) {
//         super({ name: `pca-${randomString()}`, ...config })
//         this.outputDim = config.outputDim
//         this.epsilon = config.epsilon || 1e-7
//         this.powerIterations = config.powerIterations || 10
//     }

//     build(inputShape) {
//         const inputDim = inputShape[inputShape.length - 1]
//         this.W = this.addWeight(
//             'W',
//             [inputDim, this.outputDim],
//             'float32',
//             tf.initializers.orthogonal({ gain: 1.0 })
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             // Center the data
//             const mean = tf.mean(inputs, [0, 1], true)
//             const centered = tf.sub(inputs, mean)

//             // Compute covariance matrix
//             const [batchSize, seqLength, inputDim] = inputs.shape
//             const flattenedCentered = tf.reshape(centered, [-1, inputDim])
//             const cov = tf
//                 .matMul(flattenedCentered, flattenedCentered, true, false)
//                 .div(tf.scalar(flattenedCentered.shape[0] - 1))

//             // Compute approximation of principal components
//             const W = this.approximatePCA(cov)

//             // Project data onto principal components
//             const result = tf.matMul(flattenedCentered, W)
//             return tf.reshape(result, [batchSize, seqLength, this.outputDim])
//         })
//     }

//     approximatePCA(cov) {
//         let W = this.W.read()

//         for (let i = 0; i < this.powerIterations; i++) {
//             // Power iteration
//             W = tf.matMul(cov, W)

//             // Orthogonalize and normalize
//             W = tf.linalg.gramSchmidt(W)
//         }

//         // Update the weight variable correctly
//         this.W.write(W)

//         return W
//     }

//     computeOutputShape(inputShape) {
//         return [...inputShape.slice(0, -1), this.outputDim]
//     }

//     getWeights() {
//         return [this.W.read()]
//     }

//     setWeights(weights) {
//         this.W.write(weights[0])
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             outputDim: this.outputDim,
//             epsilon: this.epsilon,
//             powerIterations: this.powerIterations
//         }
//     }
// }

// class PrincipalComponentAnalysis extends LayerBase {
//     constructor(config) {
//         super({ name: `pca-${randomString()}`, ...config })
//         this.outputDim = config.outputDim
//         this.epsilon = config.epsilon || 1e-7
//     }

//     build(inputShape) {
//         const inputDim = inputShape[inputShape.length - 1]
//         this.W = this.addWeight(
//             'W',
//             [inputDim, this.outputDim],
//             'float32',
//             tf.initializers.orthogonal({ gain: 1.0 })
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             // Center the data
//             const mean = tf.mean(inputs, [0, 1], true)
//             const centered = tf.sub(inputs, mean)

//             // Compute covariance matrix (averaged over batch and sequence dimensions)
//             const batchSize = inputs.shape[0]
//             const seqLength = inputs.shape[1]
//             const totalSamples = batchSize * seqLength
//             const flattenedCentered = tf.reshape(centered, [totalSamples, -1])
//             const cov = tf
//                 .matMul(flattenedCentered, flattenedCentered, true, false)
//                 .div(tf.scalar(totalSamples - 1))

//             // Compute approximation of principal components
//             const W = this.approximatePCA(cov)

//             // Project data onto principal components
//             const flattenedResult = tf.matMul(flattenedCentered, W)
//             const result = tf.reshape(flattenedResult, [
//                 batchSize,
//                 seqLength,
//                 this.outputDim
//             ])

//             return result
//         })
//     }

//     approximatePCA(cov) {
//         let W = this.W.read()

//         // Power iteration method
//         for (let i = 0; i < 5; i++) {
//             W = tf.matMul(cov, W)
//             // Normalize columns
//             const norms = tf.sqrt(tf.sum(tf.square(W), 0)).add(this.epsilon)
//             W = tf.div(W, norms)
//         }

//         return W
//     }

//     computeOutputShape(inputShape) {
//         return [...inputShape.slice(0, -1), this.outputDim]
//     }

//     getWeights() {
//         return [this.W.read()]
//     }

//     setWeights(weights) {
//         this.W.write(weights[0])
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             outputDim: this.outputDim,
//             epsilon: this.epsilon
//         }
//     }
// }

// class PrincipalComponentAnalysis extends LayerBase {
//     constructor(config) {
//         super({ name: `pca-${randomString()}`, ...config })
//         this.outputDim = config.outputDim
//         this.epsilon = config.epsilon || 1e-7
//         this.maxIterations = config.maxIterations || 100
//         this.convergenceThreshold = config.convergenceThreshold || 1e-4
//     }

//     build(inputShape) {
//         const inputDim = inputShape[inputShape.length - 1]
//         this.W = this.addWeight(
//             'W',
//             [inputDim, this.outputDim],
//             'float32',
//             tf.initializers.glorotNormal()
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             // Standardize the data (center and scale)
//             const mean = tf.mean(inputs, [0, 1], true)
//             const variance = tf.mean(
//                 tf.square(tf.sub(inputs, mean)),
//                 [0, 1],
//                 true
//             )
//             const stdDev = tf.sqrt(variance.add(this.epsilon))
//             const standardized = tf.div(tf.sub(inputs, mean), stdDev)

//             // Compute covariance matrix
//             const batchSize = inputs.shape[0]
//             const seqLength = inputs.shape[1]
//             const totalSamples = batchSize * seqLength
//             const flattenedStandardized = tf.reshape(standardized, [
//                 totalSamples,
//                 -1
//             ])
//             const cov = tf
//                 .matMul(
//                     flattenedStandardized,
//                     flattenedStandardized,
//                     true,
//                     false
//                 )
//                 .div(tf.scalar(totalSamples - 1))

//             // Compute principal components
//             const [eigenvectors, eigenvalues] = this.computeEigenvectors(cov)

//             // Project data onto principal components
//             const flattenedResult = tf.matMul(
//                 flattenedStandardized,
//                 eigenvectors
//             )
//             const result = tf.reshape(flattenedResult, [
//                 batchSize,
//                 seqLength,
//                 this.outputDim
//             ])

//             return result
//         })
//     }

//     computeEigenvectors(cov) {
//         const inputDim = cov.shape[0]
//         let eigenvectors = []
//         let eigenvalues = []

//         for (let i = 0; i < this.outputDim; i++) {
//             // Initialize a random vector
//             let vector = tf.randomNormal([inputDim, 1])
//             let prevEigenvalue = 0

//             for (let iter = 0; iter < this.maxIterations; iter++) {
//                 // Power iteration
//                 vector = tf.matMul(cov, vector)

//                 // Normalize the vector
//                 const norm = tf.norm(vector)
//                 vector = tf.div(vector, norm)

//                 // Compute the Rayleigh quotient (eigenvalue estimate)
//                 const eigenvalue = tf
//                     .sum(tf.mul(tf.matMul(vector, vector, true), cov))
//                     .squeeze()

//                 // Check for convergence
//                 if (
//                     Math.abs(eigenvalue.dataSync()[0] - prevEigenvalue) <
//                     this.convergenceThreshold
//                 ) {
//                     break
//                 }
//                 prevEigenvalue = eigenvalue.dataSync()[0]
//             }

//             eigenvectors.push(vector)
//             eigenvalues.push(prevEigenvalue)

//             // Deflate the covariance matrix
//             const projection = tf.matMul(vector, vector, true)
//             cov = tf.sub(cov, tf.mul(projection, prevEigenvalue))
//         }

//         // Combine eigenvectors into a matrix
//         const eigenvectorMatrix = tf.concat(eigenvectors, 1)

//         // Update the layer weights
//         this.W.write(eigenvectorMatrix)

//         return [eigenvectorMatrix, tf.tensor(eigenvalues)]
//     }

//     computeOutputShape(inputShape) {
//         return [...inputShape.slice(0, -1), this.outputDim]
//     }

//     getWeights() {
//         return [this.W.read()]
//     }

//     setWeights(weights) {
//         this.W.write(weights[0])
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             outputDim: this.outputDim,
//             epsilon: this.epsilon,
//             maxIterations: this.maxIterations,
//             convergenceThreshold: this.convergenceThreshold
//         }
//     }
// }

const exportedLayers = [
    AdaptiveMixtureOfExperts,
    Antirectifier,
    AttentionFreeTransformer,
    Autoencoder,
    Bias,
    CapsNet,
    CausalSelfAttention,
    ChunkedStateSpace,
    CollapseOneHot,
    Collideoscope,
    CompressorHead,
    ControlGate,
    ConvolutionalExpansionLayer,
    DebugLayer,
    DenseMultiLayerPerceptron,
    DepthwiseSeparableConvolution,
    DeterministicEmbedding,
    DimensionContraction,
    DimensionExpansion,
    DumbCompression,
    EfficientAttention,
    EfficientChannelAttention,
    FastAssociativeMemory,
    FastMemory,
    FourierFeaturePositionalEncoding,
    GatedLinearMLP,
    GroupedQueryAttention,
    IncrementalPowerIterationPCA,
    InstanceNormalization,
    Interrogator,
    KANLayer,
    KolmogorovArnoldNetwork,
    LambdaLayer,
    LazyMixtureOfExperts,
    LearnedUpsampling,
    LinearAttention,
    LocalSelfAttention,
    MeltingMLP,
    MixtureOfDepthsRouting,
    MixtureOfExperts,
    MultiHeadAttention,
    MultiHeadMoeBlock,
    MultiLayerPerceptron,
    MultiQueryAttention,
    NearestNeighborUpsampling,
    NystromAttention,
    OuroboticMemory,
    PerformerAttention,
    PseudoQuantumState,
    QuantumStateMachine,
    Range,
    ResidualConnection,
    RotaryPositionalEncoding,
    SelfAttention,
    SequenceExpansionLayer,
    SharedEmbedding,
    SinusoidalPositionalEncoding,
    SparseEstimatedAttention,
    SparseMixtureOfExperts,
    SqueezeAndExcitation,
    StatePlacement,
    StateSpace,
    SynthesizerAttention,
    TemporalPooling,
    ToOneHot,
    TransientMixtureOfExperts,
    WeightedSum
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})
