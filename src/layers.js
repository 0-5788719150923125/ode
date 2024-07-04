import * as tf from '@tensorflow/tfjs'
import customOps from './ops.js'
import customActivations from './activations.js'
import { randomString } from './utils.js'

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

// Loosely-inspired by Performer:
// https://arxiv.org/abs/2009.14794
class RandomFeatureAttention extends LayerBase {
    constructor(config) {
        super({ name: `attn-${randomString()}`, ...config })
        this.hiddenDim = config.hiddenDim || 256
        this.numFeatures = config.numFeatures || 256
        this.numHeads = config.numHeads || 8
        this.headDim = Math.floor(this.hiddenDim / this.numHeads)
        this.eps = 1e-6
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.inputDim = inputDim

        // Create weight matrices for each head
        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []
        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.1 })
                )
            )
        }

        this.outputKernel = this.addWeight(
            `outputKernel`,
            [this.hiddenDim, inputDim],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.1 })
        )
        this.residual = customLayers.ResidualConnection()

        this.randomMatrix = tf.randomNormal(
            [this.headDim, this.numFeatures],
            0,
            1 / Math.sqrt(this.numFeatures)
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            // Ensure inputs is a tensor, not an array
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Process each head
            const headOutputs = this.queryKernels.map((queryKernel, i) => {
                // Linear transformations to create query, key, and value for this head
                const Q = this.applyDense(inputs, queryKernel.read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                // Apply random feature map to query and key
                const QF = this.applyFeatureMap(Q)
                const KF = this.applyFeatureMap(K)

                // Compute key-value representation
                const KFV = tf.matMul(KF, V, true, false)
                // Compute normalization factor
                const D = tf.sum(KF, -2, true)

                // Compute attention scores
                const QF_KFV = tf.matMul(QF, KFV)

                // Compute normalization term via element-wise multiplication for efficient broadcasting
                const QF_D = tf.mul(QF, D)
                // Sum over the feature dimension
                const QF_D_sum = tf.sum(QF_D, -1, true)

                // Implementation of attention mechanism with epsilon for numerical stability
                return tf.div(QF_KFV, tf.add(QF_D_sum, this.eps))
            })

            // Concatenate head outputs
            let outputs = tf.concat(headOutputs, -1)

            // Apply layer normalization
            outputs = this.rmsNorm(outputs)
            // Apply output projection
            outputs = this.applyDense(outputs, this.outputKernel.read())
            // Scale down outputs for stability
            outputs = tf.mul(outputs, tf.scalar(0.1))

            // Apply residual connection
            return this.residual.apply([inputs, outputs])
        })
    }

    applyFeatureMap(x) {
        const projection = tf.matMul(x, this.randomMatrix)
        // ReLU activation for sparsity and efficiency
        return tf.relu(projection)
    }

    getWeights() {
        return [
            ...this.queryKernels.map((k) => k.read()),
            ...this.keyKernels.map((k) => k.read()),
            ...this.valueKernels.map((k) => k.read()),
            this.outputKernel.read(),
            this.randomMatrix
        ]
    }

    setWeights(weights) {
        const headWeights = weights.slice(0, -1)
        const numHeadWeights = headWeights.length
        const weightsPerHead = numHeadWeights / 3

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels[i].write(headWeights[i])
            this.keyKernels[i].write(headWeights[i + weightsPerHead])
            this.valueKernels[i].write(headWeights[i + 2 * weightsPerHead])
        }

        this.outputKernel.write(weights[weights.length - 2])
        this.randomMatrix.write(weights[weights.length - 1])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            numFeatures: this.numFeatures,
            numHeads: this.numHeads
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

// A chunk-based approach leads to O(n * chunk_size) memory, which is
// linear if chunk size is fixed.
// TODO: sliding window and heirarchical versions of this
class LocalSelfAttention extends LayerBase {
    constructor(config) {
        super({ name: `attn-${randomString()}`, ...config })
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

            const Q = this.applyDense(inputs, this.queryKernel.read())
            const K = this.applyDense(inputs, this.keyKernel.read())
            const V = this.applyDense(inputs, this.valueKernel.read())

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
                    this.queryKernels[i].read(),
                    this.queryBiases[i].read()
                )
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

            const K = this.applyDense(
                inputs,
                this.keyKernel.read(),
                this.keyBias.read()
            )
            const V = this.applyDense(
                inputs,
                this.valueKernel.read(),
                this.valueBias.read()
            )

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.queries; i++) {
                const Q = this.applyDense(
                    inputs,
                    this.queryKernels[i].read(),
                    this.queryBiases[i].read()
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
                this.outputKernel.read(),
                this.outputBias.read()
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

class VariableDimensionMLP extends LayerBase {
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
                this.inProjBias.read()
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

class AdaptiveMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.topK = config.topK || 2
        this.switchingDim = config?.switchingDim || 64
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

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
            [this.topK * inputDim, inputDim],
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

            const expertIndices = this.selectTopExperts(weightedAvgScores)

            // Predict on top-k experts, for every batch
            const batchOutputs = []
            for (let i = 0; i < inputs.shape[0]; i++) {
                const batchInputs = inputs.slice([i, 0], [1, -1])
                const expertOutputs = []
                for (let j = 0; j < this.topK; j++) {
                    const expertIndex = expertIndices[i][j]
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

    selectTopExperts(switchingScores) {
        const topKIndices = tf.topk(switchingScores, this.topK).indices
        return topKIndices.arraySync()
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
            topK: this.topK
        }
    }
}

// https://arxiv.org/abs/2404.02258
class MixtureOfDepths extends LayerBase {
    constructor(config) {
        super({ name: `mod-${randomString()}`, ...config })
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.capacity = config.capacity || 0.125
        this.temperature = config.temperature || 0.1
        this.auxLossWeight = config.auxLossWeight || 0.01
        this.emaDecay = config.emaDecay || 0.99
        this.expertUsageEMA = null
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.routerKernel = this.addWeight(
            'routerKernel',
            [inputDim, 1],
            'float32',
            tf.initializers.varianceScaling({
                scale: 0.01,
                distribution: 'normal',
                mode: 'fanAvg'
            })
        )
        this.routerBias = this.addWeight(
            'routerBias',
            [1],
            'float32',
            tf.initializers.zeros()
        )
        this.expertUsageEMA = tf.variable(tf.zeros([this.numExperts]), false)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, timeSteps, inputDim] = inputs.shape

            // Router network
            const routerLogits = this.applyDense(
                inputs,
                this.routerKernel.read(),
                this.routerBias.read()
            ).reshape([batchSize, timeSteps])

            return tf.customGrad((x, save) => {
                // Forward pass: Top-k selection
                const k = Math.floor(this.capacity * timeSteps)
                const { values: topKValues, indices: topKIndices } = tf.topk(
                    routerLogits,
                    k
                )

                const topkMask = tf
                    .oneHot(topKIndices, timeSteps)
                    .sum(1)
                    .expandDims(-1)

                // Apply top-k mask to inputs
                const selectedTokens = x.mul(topkMask)
                const residualTokens = x.mul(
                    tf.onesLike(topkMask).sub(topkMask)
                )

                // Apply layer to routed tokens
                let selectedOutputs = selectedTokens
                for (const expert of this.experts) {
                    selectedOutputs = expert.apply(selectedOutputs)
                }

                // Combine processed tokens with residual tokens
                const output = selectedOutputs.add(residualTokens)

                const savedTensors = [routerLogits, topkMask, x]

                // Compute auxiliary loss
                if (kwargs.training) {
                    savedTensors.push(
                        this.computeAuxLoss(topKValues, topKIndices)
                    )
                }

                save(savedTensors)

                // Define gradient function
                const gradFunc = (dy, saved) => {
                    const [routerLogits, topkMask, originalInputs] =
                        saved.slice(0, 2)

                    let auxLoss
                    if (kwargs.training) {
                        auxLoss = saved[3]
                    }

                    // Backward pass: Gumbel-Softmax
                    const gumbelMask = this.ode.ops
                        .gumbelSoftmax(routerLogits, this.temperature)
                        .expandDims(-1)

                    // Compute gradients for the selected tokens
                    let selectedGrads = dy.mul(gumbelMask)
                    for (const expert of this.experts) {
                        selectedGrads = expert.apply(selectedGrads)
                    }

                    // Compute gradients for the residual tokens
                    const residualGrads = dy.mul(
                        tf.onesLike(gumbelMask).sub(gumbelMask)
                    )

                    // Combine the selected and residual gradients
                    let inputGrads = selectedGrads.add(residualGrads)

                    // Add auxiliary loss gradient
                    if (kwargs.training) {
                        inputGrads = inputGrads.add(
                            auxLoss.mul(this.auxLossWeight)
                        )
                    }

                    return inputGrads
                }

                return { value: output, gradFunc }
            })(inputs)
        })
    }

    computeAuxLoss(topKIndices) {
        return tf.tidy(() => {
            const [batchSize, k] = topKIndices.shape
            const numExperts = this.numExperts

            // Compute current expert usage
            const currentUsage = tf
                .oneHot(topKIndices.cast('int32'), numExperts)
                .sum([0, 1])
                .div(tf.scalar(batchSize * k))

            // Update EMA
            const newEMA = this.expertUsageEMA
                .mul(this.emaDecay)
                .add(currentUsage.mul(1 - this.emaDecay))
            this.expertUsageEMA.assign(newEMA)

            // Compute load balancing loss
            const idealUsage = tf.ones([numExperts]).div(numExperts)
            const loadBalancingLoss = tf
                .squaredDifference(this.expertUsageEMA, idealUsage)
                .mean()

            // Compute expert utilization loss
            const utilizationLoss = tf
                .log(this.expertUsageEMA.add(1e-5))
                .neg()
                .mean()

            // Combine losses
            const loss = loadBalancingLoss.add(utilizationLoss)
            // console.log(loss.dataSync()[0])
            return loss
        })
    }

    getWeights() {
        return [this.routerKernel.read(), this.routerBias.read()]
    }

    setWeights(weights) {
        this.routerKernel.write(weights[0])
        this.routerBias.write(weights[1])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            capacity: this.capacity,
            temperature: this.temperature,
            auxLossWeight: this.auxLossWeight,
            emaDecay: this.emaDecay
        }
    }
}

class SwarmOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.hiddenDim = config.hiddenDim || 64
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

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

            // Compute expert outputs
            const expertOutputs = this.experts.map((expert, i) =>
                this.experts[i].apply(inputs)
            )

            // Combine expert outputs using weighted sum
            const combinedOutput = expertOutputs.reduce((prev, curr, i) => {
                const expertWeight = expertWeights.slice(
                    [0, 0, i],
                    [inputs.shape[0], inputs.shape[1], 1]
                )
                return prev.add(curr.mul(expertWeight))
            }, tf.zeros(expertOutputs[0].shape))

            return combinedOutput
        })
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

// https://arxiv.org/abs/2306.03745
class SMEARMoE extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.hiddenDim = config.hiddenDim || 64
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        // Initialize gating network
        this.routerHiddenKernel = this.addWeight(
            'routerHiddenKernel',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerHiddenBias = this.addWeight(
            'routerHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerOutputKernel = this.addWeight(
            'routerOutputKernel',
            [this.hiddenDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerOutputBias = this.addWeight(
            'routerOutputBias',
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
                this.routerHiddenKernel.read(),
                this.routerHiddenBias.read()
            )

            // Apply layer normalization before activating the logits of our router
            const normalizedState = this.rmsNorm(gatingHidden)
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(normalizedState)

            const expertWeights = this.applyDense(
                activatedGate,
                this.routerOutputKernel.read(),
                this.routerOutputBias.read()
            )

            // Apply softmax to get the final routing probabilities
            const routingProbabilities = expertWeights.softmax()

            // Merge experts
            const mergedExpert = this.mergeExperts(
                this.experts,
                routingProbabilities
            )

            // Pass inputs to merged expert
            return mergedExpert.apply(inputs)
        })
    }

    mergeExperts(experts, weights) {
        // We modify the first expert in-place
        const mergedExpert = experts[0]
        // We skip the first expert during averaging
        const usedExperts = experts.slice(1)

        // Aggregate weights across batch and sequence dimensions
        const aggregatedWeights = weights.sum([0, 1]) // Shape: [num_experts]

        // Normalize the aggregated weights
        const normalizedWeights = aggregatedWeights.div(aggregatedWeights.sum())

        for (let i = 0; i < mergedExpert.layers.length; i++) {
            const layerWeights = mergedExpert.layers[i].getWeights()

            // Compute weighted average of weights for this layer across all experts
            const averagedWeights = layerWeights.map((_, weightIndex) => {
                const expertWeights = usedExperts.map(
                    (expert) => expert.layers[i].getWeights()[weightIndex]
                )

                const weightedAverage = tf.tidy(() => {
                    return expertWeights.reduce((sum, weight, expertIndex) => {
                        const expertWeight = normalizedWeights.slice(
                            [expertIndex],
                            [1]
                        )
                        const weightedExpert = weight.mul(expertWeight)

                        return sum.add(weightedExpert)
                    }, tf.zeros(expertWeights[0].shape))
                })

                return weightedAverage
            })

            // Set the averaged weights to the merged expert's layer
            mergedExpert.layers[i].setWeights(averagedWeights)
        }

        return mergedExpert
    }

    getWeights() {
        return [
            this.routerHiddenKernel.read(),
            this.routerHiddenBias.read(),
            this.routerOutputKernel.read(),
            this.routerOutputBias.read()
        ]
    }

    setWeights(weights) {
        this.routerHiddenKernel.write(weights[0])
        this.routerHiddenBias.write(weights[1])
        this.routerOutputKernel.write(weights[2])
        this.routerOutputBias.write(weights[3])
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

            let hInitial = this.applyDense(inputs, this.C.read(), this.b.read())

            hInitial = hInitial.add(this.applyDense(this.hPrev, this.W.read()))

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

class CapsNet extends LayerBase {
    constructor(config) {
        super({ name: `cap-${randomString()}`, ...config })
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.numCapsules = config?.numCapsules || 8
        this.capsuleDim = config?.capsuleDim || 16
        this.routingIterations = config?.routingIterations || 3
        this.activation = config?.activation || 'relu'
        this.supportsMasking = true
    }

    build(inputShape) {
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
            [this.numCapsules * this.capsuleDim, this.units],
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

        // Initialize weights for primary capsules
        this.primaryCapsKernel = this.addWeight(
            'primaryCapsKernel',
            [this.innerDim, this.numCapsules * this.capsuleDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.primaryCapsBias = this.addWeight(
            'primaryCapsBias',
            [this.numCapsules * this.capsuleDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        this.digitCaps = new DigitCaps({
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            routingIterations: this.routingIterations
        })

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const inputShape = inputs.shape
            const batchSize = inputShape[0]
            const sequenceLength = inputShape[1]

            // Expand and contract projection via feedforward layers
            let outputs = this.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias.read()
            )
            // Activate inputs
            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)
            // Apply layer norm
            outputs = this.rmsNorm(outputs)
            // Apply primary capsules
            outputs = this.applyDense(
                outputs,
                this.primaryCapsKernel.read(),
                this.primaryCapsBias.read()
            )

            // Reshape for primary capsules
            outputs = tf.reshape(outputs, [
                batchSize * sequenceLength,
                this.numCapsules,
                this.capsuleDim
            ])

            // Apply digit capsules with dynamic routing
            outputs = this.digitCaps.apply(outputs)

            // Reshape back to original sequence shape
            outputs = tf.reshape(outputs, [
                batchSize,
                sequenceLength,
                this.numCapsules * this.capsuleDim
            ])

            outputs = this.applyDense(
                outputs,
                this.outProjKernel.read(),
                this.outProjBias.read()
            )

            // If training, apply residual dropout
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs
            // Apply skip connection
            return this.residual.apply([inputs, outputs])
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getWeights() {
        return [
            this.inProjKernel.read(),
            this.inProjBias.read(),
            this.primaryCapsKernel.read(),
            this.primaryCapsBias.read(),
            this.outProjKernel.read(),
            this.outProjBias.read(),
            ...this.digitCaps.getWeights()
        ]
    }

    setWeights(weights) {
        this.inProjKernel.write(weights[0])
        this.inProjBias.write(weights[1])
        this.primaryCapsKernel.write(weights[2])
        this.primaryCapsBias.write(weights[3])
        this.outProjKernel.write(weights[4])
        this.outProjBias.write(weights[5])
        this.digitCaps.setWeights(weights.slice(6))
    }

    getConfig() {
        return {
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            activation: this.activation,
            routingIterations: this.routingIterations
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

    getWeights() {
        return [this.W.read()]
    }

    setWeights(weights) {
        this.W.write(weights[0])
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
                this.cAttnKernel.read(),
                this.cAttnBias.read()
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
            outputs = this.applyDense(
                outputs,
                this.cProjKernel.read(),
                this.cProjBias.read()
            )
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

class IndependentComponentAnalysis extends LayerBase {
    constructor(config) {
        super({ name: `ica-${randomString()}`, ...config })
        this.outputDim = config.outputDim
        this.maxIterations = config.maxIterations || 10
        this.maxPowerIterations = config.maxPowerIterations || 100
        this.tolerance = config.tolerance || 1e-6
        this.activation = config.activation || 'softsign'
    }

    build(inputShape) {
        this.inputDim = inputShape[inputShape.length - 1]
        this.kernelShape = [this.outputDim, this.inputDim]
        this.kernel = this.addWeight(
            'kernel',
            this.kernelShape,
            'float32',
            tf.initializers.glorotNormal()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = inputs[0]
            const [batchSize, seqLength, featureDim] = input.shape

            // Reshape to 2D for processing
            const reshapedInput = input.reshape([-1, featureDim])

            const centered = tf.sub(reshapedInput, tf.mean(reshapedInput, 0))
            const whitened = this.whiten(centered)
            const ica = this.fastICA(whitened)
            const output = tf.matMul(whitened, ica.transpose())

            // Reshape back to 3D
            return output.reshape([batchSize, seqLength, this.outputDim])
        })
    }

    whiten(X) {
        const { u, s } = this.approximateSVD(X)
        const whitened = tf.matMul(X, u)
        const epsilon = 1e-8
        const scaled = tf.div(whitened, tf.sqrt(tf.maximum(s, epsilon)))
        return scaled
    }

    approximateSVD(X) {
        const { mean, stdev } = tf.moments(X, 0)
        const centered = tf.sub(X, mean)
        const covMatrix = tf
            .matMul(centered.transpose(), centered)
            .div(X.shape[0] - 1)
        const { eigenvectors, eigenvalues } = this.powerIteration(
            covMatrix,
            this.inputDim,
            this.maxPowerIterations
        )
        return { u: eigenvectors, s: eigenvalues }
    }

    powerIteration(A, numEigenvectors, maxIterations = 100, tolerance = 1e-6) {
        let Q = tf.randomNormal([A.shape[0], numEigenvectors])
        let Qprev = Q
        let eigenvalues = tf.zeros([numEigenvectors])

        for (let i = 0; i < maxIterations; i++) {
            Q = tf.matMul(A, Q)
            const norms = tf.sqrt(tf.sum(tf.square(Q), 0))
            Q = tf.div(Q, norms)

            eigenvalues = tf.sum(tf.mul(Q, tf.matMul(A, Q)), 0)

            const diff = tf.mean(tf.abs(tf.sub(Q, Qprev)))
            if (diff.bufferSync().get(0) < tolerance) {
                break
            }
            Qprev = Q
        }

        return { eigenvectors: Q, eigenvalues }
    }

    fastICA(X) {
        let W = tf.randomNormal([this.outputDim, X.shape[1]])

        for (let i = 0; i < this.maxIterations; i++) {
            const Wprev = W

            W = tf.tidy(() => {
                const WX = tf.matMul(W, X.transpose())
                const G = tf.layers
                    .activation({ activation: this.activation })
                    .apply(WX)
                const Gder = tf.sub(
                    1,
                    tf.square(
                        tf.layers
                            .activation({ activation: this.activation })
                            .apply(WX)
                    )
                )
                const GX = tf.matMul(G, X)
                const newW = tf.sub(
                    GX.div(X.shape[0]),
                    tf.mean(Gder, 1, true).mul(W)
                )

                // Orthogonalize W
                const { u } = this.approximateSVD(newW)
                return tf.matMul(
                    u.slice([0, 0], [this.outputDim, this.outputDim]),
                    W
                )
            })

            const distanceW = tf.mean(
                tf.abs(tf.sub(W, Wprev.slice([0, 0], W.shape)))
            )
            if (distanceW.bufferSync()[0] < this.tolerance) {
                break
            }
        }

        return W
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getWeights() {
        return [this.kernel.read()]
    }

    setWeights(weights) {
        this.kernel.write(weights[0])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim,
            maxIterations: this.maxIterations,
            maxPowerIterations: this.maxPowerIterations,
            tolerance: this.tolerance,
            activation: this.activation
        }
    }
}

const exportedLayers = [
    IndependentComponentAnalysis,
    AdaptiveMixtureOfExperts,
    Antirectifier,
    AttentionFreeTransformer,
    Autoencoder,
    Bias,
    CapsNet,
    CausalSelfAttention,
    ChunkedStateSpace,
    DeterministicEmbedding,
    EfficientAttention,
    EfficientChannelAttention,
    FastAssociativeMemory,
    FourierFeaturePositionalEncoding,
    GatedLinearMLP,
    GroupedQueryAttention,
    IncrementalPowerIterationPCA,
    LambdaLayer,
    LinearAttention,
    RandomFeatureAttention,
    LocalSelfAttention,
    VariableDimensionMLP,
    MixtureOfDepths,
    MixtureOfExperts,
    MultiHeadAttention,
    MultiHeadMoeBlock,
    MultiLayerPerceptron,
    MultiQueryAttention,
    OuroboticMemory,
    Range,
    ResidualConnection,
    RotaryPositionalEncoding,
    SelfAttention,
    SharedEmbedding,
    SinusoidalPositionalEncoding,
    SMEARMoE,
    SparseMixtureOfExperts,
    SqueezeAndExcitation,
    StateSpace,
    SwarmOfExperts,
    SynthesizerAttention,
    WeightedSum
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})
