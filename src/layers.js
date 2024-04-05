import * as tf from '@tensorflow/tfjs'
import { GELU } from './activations.js'
import { randomString, seededPRNG, seededValueFromArray } from './utils.js'

const customLayers = {
    dense: (config) =>
        tf.layers.dense({ ...config, name: `ffd-${randomString()}` }),
    embedding: (config) =>
        tf.layers.embedding({ ...config, name: `emb-${randomString()}` }),
    input: (config) =>
        tf.layers.input({ ...config, name: `inp-${randomString()}` })
}
export default customLayers

class LayerBase extends tf.layers.Layer {
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
            // stuff should go here
        })
    }

    static get className() {
        return 'LayerBase'
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
            this.invokeCallHook(inputs, kwargs)
            console.log(inputs)
            inputs.print()
            console.log(inputs.dataSync())
            console.log(inputs.shape)
            return inputs
        })
    }

    static get className() {
        return 'DebugLayer'
    }
}

class Range extends LayerBase {
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

    static get className() {
        return 'Range'
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

    call(inputs, kwargs, training) {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0]
            }
            this.invokeCallHook(inputs, kwargs)

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

            const dense = (x, kernel, bias) => {
                const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
                const m = tf.matMul(x, k)
                if (this.bias) {
                    return tf.add(m, bias.read())
                } else {
                    return m
                }
            }

            const cAttn = dense(inputs, this.cAttnKernel, this.cAttnBias)

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
            outputs = dense(outputs, this.cProjKernel, this.cProjBias)
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            outputs = this.layerNorm.apply(outputs)
            return tf.layers.add().apply([inputs, outputs])
        })
    }

    static get className() {
        return 'CausalSelfAttention'
    }
}

class SinusoidalPositionalEncoding extends LayerBase {
    constructor({ units, reverse = false }) {
        super()
        this.units = units // Dimensionality of the positional encoding
        this.reverse = reverse // Flag to toggle the order of sine and cosine
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Determine the sequence length from the input shape
            const seqLength = inputs.shape[1]

            // Compute the positional encodings (2D tensor of shape [seqLength, this.units])
            const positionalEncoding = tf.tensor2d(
                Array.from({ length: seqLength }, (_, pos) => {
                    return Array.from({ length: this.units }, (_, i) => {
                        const divTerm = Math.pow(
                            10000,
                            (2 * (i / 2)) / this.units
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
            return positionalEncoding
        })
    }

    computeOutputShape(inputShape) {
        // Input shape is [batch_size, sequence_length]
        // Output shape is [batch_size, sequence_length, this.units]
        return [inputShape[0], inputShape[1], this.units]
    }

    static get className() {
        return 'SinusoidalPositionalEncoding'
    }
}

class MultiHeadAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.heads = config.heads
        this.units = config.units
        this.epsilon = config.epsilon || 1e-5
        this.depth = this.units / this.heads
        this.useCausalMask = config.useCausalMask || true
    }

    build(inputShape) {
        this.queryDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: 'glorotUniform'
        })
        this.queryDense.build(inputShape)
        this._trainableWeights.push(...this.queryDense.trainableWeights)

        this.keyDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: 'glorotUniform'
        })
        this.keyDense.build(inputShape)
        this._trainableWeights.push(...this.keyDense.trainableWeights)

        this.valueDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: 'glorotUniform'
        })
        this.valueDense.build(inputShape)
        this._trainableWeights.push(...this.valueDense.trainableWeights)

        this.outputDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: 'glorotUniform'
        })
        this.outputDense.build(inputShape)
        this._trainableWeights.push(...this.outputDense.trainableWeights)

        this.layerNorm = tf.layers.layerNormalization({ epsilon: this.epsilon })
        this.layerNorm.build(inputShape)
        this._trainableWeights.push(...this.layerNorm.trainableWeights)

        super.build(inputShape)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const training = kwargs['training'] || false
            let q = this.queryDense.apply(inputs)
            let k = this.keyDense.apply(inputs)
            let v = this.valueDense.apply(inputs)

            let batchSize = inputs.shape[0]
            let seqLength = inputs.shape[1]
            q = tf
                .reshape(q, [batchSize, -1, this.heads, this.depth])
                .transpose([0, 2, 1, 3])
            k = tf
                .reshape(k, [batchSize, -1, this.heads, this.depth])
                .transpose([0, 2, 3, 1])
            v = tf
                .reshape(v, [batchSize, -1, this.heads, this.depth])
                .transpose([0, 2, 1, 3])

            let attentionScores = tf
                .matMul(q, k)
                .div(tf.sqrt(tf.scalar(this.depth)))

            if (this.useCausalMask) {
                const causalMask = tf.linalg
                    .bandPart(tf.ones([seqLength, seqLength]), -1, 0)
                    .expandDims(0)
                    .expandDims(1)
                attentionScores = tf.where(
                    tf.equal(causalMask, 1),
                    attentionScores,
                    tf.fill(attentionScores.shape, -1e9)
                )
            }

            let attentionWeights = tf.softmax(attentionScores, -1)
            if (training) {
                // Apply dropout to attentionWeights if training is true
            }
            let attentionOutput = tf
                .matMul(attentionWeights, v)
                .transpose([0, 2, 1, 3])
                .reshape([batchSize, -1, this.units])

            let output = this.outputDense.apply(attentionOutput)
            output = tf.add(output, inputs) // Apply residual connection

            return this.layerNorm.apply(output)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            heads: this.heads,
            units: this.units,
            useCausalMask: this.useCausalMask
        }
    }

    static get className() {
        return 'MultiHeadAttention'
    }
}

class MultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super({ ...config, name: `mlp-${randomString()}` })
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || 1e-5
        if (config?.activation === 'gelu') {
            this.customActivation = new GELU()
            this.activation = 'linear'
        } else {
            this.activation = config?.activation || 'relu'
        }
        this.supportsMasking = true
    }

    build(inputShape) {
        // Initialize dense layers for projection
        this.inProj = tf.layers.dense({
            units: this.innerDim,
            inputDim: this.units,
            activation: this.activation
        })
        // We shouldn't use 2 dense layers here, since kernel names will conflict during serialization to disk
        this.outProj = tf.layers.dense({
            units: this.units,
            inputDim: inputShape,
            activation: 'linear'
        })

        // Manually call build on dense layers to initialize weights
        this.inProj.build(inputShape)
        this.outProj.build([inputShape[0], this.innerDim])

        // Initialize layer normalization
        this.layernorm = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layernorm.build(inputShape)

        // Collect all trainable weights from internal layers
        this._trainableWeights = [
            ...this.inProj.trainableWeights,
            ...this.outProj.trainableWeights,
            ...this.layernorm.trainableWeights
        ]

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
            // Expand and contract projection via feedfoward layers
            let outputs = this.inProj.apply(inputs)
            if (this.customActivation) {
                outputs = this.customActivation.apply(outputs)
            }
            outputs = this.outProj.apply(outputs)
            // If training, apply residual dropout
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs
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
            dropout: this.dropout
        }
    }

    static get className() {
        return 'MultiLayerPerceptron'
    }
}

class GatedLinearUnit extends MultiLayerPerceptron {
    constructor(config) {
        super({ ...config, name: `glu-${randomString()}` })
    }

    build(inputShape) {
        // Initialize dense layers for projection and gating
        this.inProj = tf.layers.dense({
            units: this.innerDim,
            inputDim: this.units,
            activation: this.activation
        })
        this.gateProj = tf.layers.dense({
            units: this.innerDim,
            inputDim: this.units,
            activation: 'sigmoid'
        })
        this.outProj = tf.layers.dense({
            units: this.units,
            inputDim: inputShape,
            activation: 'linear'
        })

        // Manually call build on dense layers to initialize weights
        this.inProj.build(inputShape)
        this.gateProj.build(inputShape)
        this.outProj.build([inputShape[0], this.innerDim])

        // Initialize layer normalization
        this.layernorm = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layernorm.build(inputShape)

        // Collect all trainable weights from internal layers
        this._trainableWeights = [
            ...this.inProj.trainableWeights,
            ...this.gateProj.trainableWeights,
            ...this.outProj.trainableWeights,
            ...this.layernorm.trainableWeights
        ]

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
            // Expand and contract projection via feedforward layers
            const proj = this.inProj.apply(inputs)
            const gate = this.gateProj.apply(inputs)
            const gatedOutput = tf.mul(proj, gate)
            let outputs = this.outProj.apply(gatedOutput)
            // If training, apply residual dropout
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs
            // Apply layer norm
            outputs = this.layernorm.apply(outputs)
            // Apply skip connection
            return this.residual.apply([inputs, outputs])
        })
    }

    static get className() {
        return 'GatedLinearUnit'
    }
}

class EncapsulatedMLP extends LayerBase {
    constructor(config) {
        super({ ...config, name: `cap-${randomString()}` })
        // Inherits most settings from the original MLP configuration
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || 1e-5
        this.activation = config?.activation || 'relu'
        this.supportsMasking = true
    }

    build(inputShape) {
        // Primary projection layer remains similar
        this.inProj = tf.layers.dense({
            units: this.innerDim,
            activation: 'linear' // Keeping linear for custom post-processing
        })

        // Capsule-like mechanism for dynamic routing
        // Note: This is conceptual; dynamic routing requires significant additional logic
        this.dynamicRouting = new DynamicRouting({
            numCapsules: this.units, // Simplification: number of capsules set to units
            dimensions: this.innerDim
            // Further configuration for dynamic routing can be added here
        })

        // Output projection adjusted for encapsulated processing
        this.outProj = tf.layers.dense({
            units: this.units,
            activation: 'linear'
        })

        // Initialize layers
        this.inProj.build(inputShape)
        this.dynamicRouting.build(inputShape)
        this.outProj.build([inputShape[0], this.innerDim])
        this.layernorm = tf.layers.layerNormalization({ epsilon: this.epsilon })
        this.layernorm.build(inputShape)

        // Collect trainable weights
        this._trainableWeights = [
            ...this.inProj.trainableWeights,
            ...this.outProj.trainableWeights,
            ...this.layernorm.trainableWeights,
            // Assuming dynamic routing has trainable parameters
            ...this.dynamicRouting.trainableWeights
        ]

        // Residual connection remains
        this.residual = new ResidualConnection()

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
            let outputs = this.inProj.apply(inputs)

            // Apply dynamic routing or capsule mechanism
            outputs = this.dynamicRouting.apply(outputs)

            outputs = this.outProj.apply(outputs)

            // Apply layer normalization and residual connection as before
            outputs = this.layernorm.apply(outputs)
            return this.residual.apply([inputs, outputs])
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            epsilon: this.epsilon
        }
    }

    static get className() {
        return 'EncapsulatedMLP'
    }
}

class DynamicRouting extends LayerBase {
    constructor(config) {
        super(config)
        this.numCapsules = config.numCapsules
        this.dimensions = config.dimensions
        // Assuming 'dimensions' represents the size of the output vectors from capsules
    }

    build(inputShape) {
        // Simplified routing weights
        this.routingWeights = this.addWeight(
            'routingWeights',
            [
                this.numCapsules,
                this.dimensions,
                inputShape[inputShape.length - 1]
            ],
            'float32',
            tf.initializers.randomNormal({ stddev: 0.1 })
        )
        super.build(inputShape)
    }

    // Assuming inputs have a shape of [batch_size, units, innerDim]
    // and you wish to apply dynamic routing in a manner that involves smaller subnetworks or "capsules"

    // First, ensure that your routing weights are initialized to match the operation.
    // If your operation needs to be [batch_size, units*innerDim] @ [units*innerDim, some_dimension],
    // then you must reshape or partition your inputs or weights accordingly.

    // Here's a conceptual adjustment focusing on smaller, manageable parts of the dynamic layer:
    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        // Flatten inputs to 2D for matMul compatibility, preserving batch information.
        const batch_size = inputs.shape[0]
        const flattenedInputs = tf.reshape(inputs, [batch_size, -1]) // Now [batch_size, units*innerDim]

        // Instead of one massive weight, consider having smaller "capsule" weights
        // For illustration, let's say we break it down into several smaller matrices
        // Adjust the number of capsules and the dimensionality as needed
        const numCapsules = this.numCapsules // e.g., much smaller like 10 or 20
        const capsuleDimension = this.dimensions // Adjust based on what each capsule should output

        // Initialize smaller capsule weights as part of the build process or here if dynamic
        // Let's simplify and not show the initialization here for brevity

        // Example loop over smaller capsule weights (assuming they are stored in an array this.capsuleWeights)
        let capsuleOutputs = []
        for (let i = 0; i < numCapsules; i++) {
            let capsuleWeight = this.capsuleWeights[i] // Each with shape [units*innerDim/numCapsules, capsuleDimension]
            let output = tf.matMul(flattenedInputs, capsuleWeight) // Outputs for each capsule
            capsuleOutputs.push(output)
        }

        // Assuming you want to concatenate or somehow aggregate these outputs
        let concatenatedOutputs = tf.concat(capsuleOutputs, 1) // Adjust axis as needed
        // Now reshape or process concatenatedOutputs as needed to fit the next layer

        return concatenatedOutputs // This needs to be reshaped or adjusted to match your model's expectations
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.numCapsules, this.dimensions]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numCapsules: this.numCapsules,
            dimensions: this.dimensions
        }
    }

    static get className() {
        return 'DynamicRouting'
    }
}

class PassthroughLayer extends LayerBase {
    constructor(config) {
        super({ ...config, name: `pass-${randomString()}` })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            console.log(inputs)
            return inputs
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig()
        }
    }

    static get className() {
        return 'PassthroughLayer'
    }
}

class SparseMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ ...config, name: `moe-${randomString()}` })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 128
        this.experts = config.experts
        this.numExperts = this.experts.length
        this.topK = config.topK || 2
        this.loadBalancing = config.loadBalancing || 1.0
        this.usageDecay = config.usageDecay || 0.99
        this.usageHistory = tf.zeros([this.numExperts])
        this.extraLoss = 0
        this.currentStep
    }

    build(inputShape) {
        this.inGate = tf.layers.dense({
            units: this.units,
            activation: 'tanh'
        })

        this.hiddenGate = tf.layers.dense({
            units: this.innerDim,
            activation: 'softsign'
        })

        this.outGate = tf.layers.dense({
            units: this.numExperts,
            activation: 'softmax'
        })

        // Build gating mechanism
        this.inGate.build(inputShape)
        const gateShape = this.inGate.computeOutputShape(inputShape)
        this.hiddenGate.build(gateShape)
        const hiddenShape = this.hiddenGate.computeOutputShape(gateShape)
        this.outGate.build(hiddenShape)

        this._trainableWeights = [
            ...this.inGate.trainableWeights,
            ...this.hiddenGate.trainableWeights,
            ...this.outGate.trainableWeights
        ]

        // Build each expert layer
        this.experts.forEach((expert) => {
            expert.build(inputShape)
            this._trainableWeights.push(...expert.trainableWeights)
        })

        super.build(inputShape)
    }

    computeGate(inputs) {
        return this.outGate.apply(
            this.hiddenGate.apply(this.inGate.apply(inputs))
        )
    }

    computeUtilization(currentExperts, gatingScores) {
        const oldUsageHistory = this.usageHistory
        tf.tidy(() => {
            // Create a zeros tensor for new usage counts.
            let newUsage = tf.zeros([this.numExperts])

            // Update the tensor based on currentExperts indices and gatingScores.
            currentExperts.forEach((expertIndex, i) => {
                const indexTensor = tf.tensor1d([expertIndex], 'int32')
                const updateTensor = tf.tensor1d([gatingScores[i]], 'float32')
                newUsage = tf
                    .scatterND(indexTensor.expandDims(0), updateTensor, [
                        this.numExperts
                    ])
                    .add(newUsage)
            })

            // Normalize the usage to proportion.
            const currentUsageProportion = newUsage.div(tf.sum(newUsage))

            // Update the usage history tensor with decay.
            this.usageHistory = oldUsageHistory
                .mul(this.usageDecay)
                .add(currentUsageProportion.mul(1 - this.usageDecay))
            tf.keep(this.usageHistory)
        })

        oldUsageHistory.dispose()

        // Calculate the dynamic penalty based on the updated usage history.
        tf.tidy(() => {
            const idealProportion = 1 / this.numExperts
            const divergences = this.usageHistory
                .sub(tf.scalar(idealProportion))
                .abs()
            this.extraLoss = divergences
                .mean()
                .mul(this.loadBalancing)
                .dataSync()[0]
        })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const gatingScores = this.computeGate(inputs)

            const topKValues = tf.topk(gatingScores, this.topK, true)

            // Sample experts from the last timestep
            const sequenceLength = inputs.shape[1]
            const currentExperts = topKValues.indices
                .slice([0, sequenceLength - 1, 0], [1, 1, this.topK])
                .reshape([this.topK])
                .arraySync()
            const currentGatingScores = topKValues.values
                .slice([0, sequenceLength - 1, 0], [1, 1, this.topK])
                .reshape([this.topK])
                .arraySync()

            this.computeUtilization(currentExperts, currentGatingScores)

            // Compute outputs for the selected experts
            const expertOutputs = currentExperts.map((index) => {
                return this.experts[index].apply(inputs)
            })

            // // Compute average of expert outputs
            // const outputSum = expertOutputs.reduce((acc, curr) => acc.add(curr))
            // const outputAverage = outputSum.div(expertOutputs.length)

            // return outputAverage

            // Compute weighted sum of expert outputs
            const outputs = expertOutputs.reduce((acc, curr, i) => {
                return acc.add(curr.mul(currentGatingScores[i]))
            }, tf.zerosLike(expertOutputs[0]))

            return outputs
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            topK: this.topK,
            loadBalancing: this.loadBalancing
        }
    }

    static get className() {
        return 'SparseMixtureOfExperts'
    }
}

class LazyMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ ...config, name: `lmoe-${randomString()}` })
        this.numRoutingIterations = config.numRoutingIterations || 3
        this.topK = config.topK || 2
        this.experts = config.experts
        this.numExperts = this.experts.length
        this.topK = Math.min(this.topK, this.numExperts)
    }

    build(inputShape) {
        this.experts.forEach((expert) => {
            expert.build(inputShape)
            this._trainableWeights.push(...expert.trainableWeights)
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

    static get className() {
        return 'LazyMixtureOfExperts'
    }
}

class ControlGate extends LayerBase {
    constructor(config) {
        super({ ...config, name: `gate-${randomString()}` })
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

        this._trainableWeights = [
            ...this.in_gate.trainableWeights,
            ...this.out_gate.trainableWeights
        ]

        // Build each expert layer
        this.experts.forEach((expert) => {
            expert.build(inputShape)
            this._trainableWeights.push(...expert.trainableWeights)
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

    static get className() {
        return 'ControlGate'
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
            this.invokeCallHook(inputs, kwargs)
            // inputs is an array where inputs[0] is the original input and inputs[1] is the output to be added to it.
            if (inputs.length !== 2) {
                throw new Error('ResidualConnection expects 2 inputs.')
            }

            const [originalInput, blockOutput] = inputs
            return tf.add(originalInput, blockOutput)
        })
    }

    static get className() {
        return 'ResidualConnection'
    }
}

class GaussianMixtureModel extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = 4
        this.hiddenSize = 256 // Assuming this is the desired hidden layer size.
        this.temperature = config.temperature || 1.0

        // Initialize weights
        this.wGate = tf.variable(
            tf.randomNormal([this.hiddenSize, this.numExperts])
        )
        this.expertCentroids = tf.variable(
            tf.randomNormal([this.numExperts, this.hiddenSize])
        )
    }

    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        const batchSize = inputs.shape[0]
        const flattenedInput = inputs.reshape([batchSize, -1])

        console.log(`flattenedInput shape: ${flattenedInput.shape}`)

        // Dynamically calculate input dimension
        const inputDim = flattenedInput.shape[1]

        // Assuming the input needs to be projected to match the hiddenSize
        const projectionMatrix = tf.variable(
            tf.randomNormal([inputDim, this.hiddenSize])
        )
        const projectedInput = flattenedInput.matMul(projectionMatrix)

        console.log(`projectedInput shape: ${projectedInput.shape}`)

        // Linear transformation with wGate
        const z = projectedInput.matMul(this.wGate)

        console.log(`z shape: ${z.shape}`)

        // Calculate log posterior probabilities
        const logits = logGmmPosterior(
            z,
            this.expertCentroids,
            this.temperature
        )

        // Example adjustment at the end of the GMM layer's call method
        return logits.reshape([batchSize, -1]) // Adjust this line based on the actual expected shape
    }

    computeOutputShape(inputShape) {
        // Output shape: batch_size x num_experts
        return [inputShape[0], this.numExperts]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenSize: this.hiddenSize,
            temperature: this.temperature
        }
    }

    static get className() {
        return 'GaussianMixtureModel'
    }
}

function logGmmPosterior(z, expertCentroids, temperature) {
    // Assuming the need to compare z (shape: [2, 4]) with centroids ([4, 256]) requires adjustment.
    // This example will adjust the operation to a more plausible comparison, given the shapes:
    // Direct multiplication isn't compatible due to the shape mismatch, suggesting a different approach is needed.

    // If expertCentroids were intended to be [numExperts, hiddenSize] = [4, 256],
    // and we need to operate on z ([batchSize, numExperts] = [2, 4]),
    // a valid operation might be to reshape or align dimensions for comparison differently.

    // Here's a placeholder operation for demonstration, adjust according to your specific logic:
    const reshapedCentroids = expertCentroids.reshape([1, 4, 256]) // Example reshape, adjust as needed.
    const expandedZ = z.expandDims(2) // Expanding z for broadcasting, adjust logic as needed.
    // Placeholder for intended comparison logic, e.g., calculating distances or similarities.

    // Assuming a simpler operation for demonstration:
    const similarity = expandedZ.mul(reshapedCentroids) // Example, adjust to match your intended logic.

    // Apply temperature scaling to the result of your actual comparison logic
    return similarity.mul(temperature)
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
    static get className() {
        return 'LambdaLayer'
    }
}

// https://arxiv.org/abs/2005.00743
// https://github.com/iafarhan/causal-synthesizer-multihead-attention/blob/main/synthesizer.py
class SynthesizerAttention extends LayerBase {
    constructor(config) {
        super({ ...config, name: `syn-${randomString()}` })
        this.units = config.units
        this.heads = config.heads
        this.blockSize = config.blockSize
        this.attnPdrop = config.dropout || 0.0
        this.residPdrop = config.dropout || 0.0
        this.activation = config.activation || tf.relu
        this.epsilon = config.epsilon || 1e-5
        this.alpha = config.alpha || 1
        this.depth = this.units / this.heads
    }

    build(inputShape) {
        this.w1 = this.addWeight(
            `w1-${randomString()}`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.w2 = this.addWeight(
            `w2-${randomString()}`,
            [this.units / this.heads, this.blockSize],
            'float32',
            tf.initializers.zeros()
        )
        this.b2 = this.addWeight(
            `b2-${randomString()}`,
            [this.blockSize],
            'float32',
            tf.initializers.zeros()
        )
        this.value = this.addWeight(
            `value-${randomString()}`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.proj = this.addWeight(
            `proj-${randomString()}`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )

        this.layernorm = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layernorm.build(inputShape)
        this._trainableWeights.push(...this.layernorm.trainableWeights)

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
            this.invokeCallHook(inputs, kwargs)
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

            const output = this.synthesize(yReshaped, this.proj.read())

            const residOutput = this.residDropout.apply(output)
            const normalized = this.layernorm.apply(residOutput)

            return this.residual.apply([inputs, normalized])
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
            residPdrop: this.dropout
        })
        return config
    }

    static get className() {
        return 'SynthesizerAttention'
    }
}

class TransformerXLAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units
        this.heads = config.heads
        this.dropout = config.dropout || 0.0
        this.epsilon = config.epsilon || 1e-6
        this.queryDim = this.units / this.heads
        this.memoryLength = config.memoryLength || 0
    }

    build(inputShape) {
        const queryDim = this.units / this.heads
        const valueDim = queryDim

        this.queryDense = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            kernelInitializer: 'glorotNormal'
        })
        this.keyDense = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            kernelInitializer: 'glorotNormal'
        })
        this.valueDense = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            kernelInitializer: 'glorotNormal'
        })
        this.relativePositionEmbedding = this.addWeight(
            'relativePositionEmbedding',
            [this.heads, this.memoryLength + inputShape[1], queryDim],
            'float32',
            tf.initializers.glorotNormal()
        )

        this.dropout = tf.layers.dropout({ rate: this.dropout })
        this.outputDense = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            kernelInitializer: 'glorotNormal'
        })

        this.layerNormalization = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layerNormalization.build(inputShape)
    }

    splitHeads(x, batchSize, numHeads, seqLength, depth) {
        const reshaped = tf.reshape(x, [batchSize, seqLength, numHeads, depth])
        return tf.transpose(reshaped, [0, 2, 1, 3])
    }

    relativePositionalEncoding(x) {
        const [batchSize, numHeads, seqLength, depth] = x.shape
        const positionEncodings = this.relativePositionEmbedding.read()
        const slicedPositionEncodings = positionEncodings.slice(
            [0, tf.backend().read(this.memoryLength), 0],
            [numHeads, seqLength, depth]
        )
        return slicedPositionEncodings
    }

    call(inputs, kwargs) {
        const x = inputs
        const [batchSize, seqLength, depth] = x.shape
        const recentMemory = kwargs['recentMemory'] || null

        const q = this.queryDense.apply(x)
        const k = this.keyDense.apply(x)
        const v = this.valueDense.apply(x)

        const queryHeads = this.splitHeads(
            q,
            batchSize,
            this.heads,
            seqLength,
            this.queryDim
        )
        const keyHeads = this.splitHeads(
            k,
            batchSize,
            this.heads,
            seqLength,
            this.queryDim
        )
        const valueHeads = this.splitHeads(
            v,
            batchSize,
            this.heads,
            seqLength,
            this.queryDim
        )

        let attention
        if (recentMemory === null) {
            const relativePositionEncodings =
                this.relativePositionalEncoding(queryHeads)
            const positionWeights = tf.einsum(
                'bhqd,bhkd->bhqk',
                queryHeads,
                relativePositionEncodings
            )
            const contentWeights = tf.einsum(
                'bhqd,bhkd->bhqk',
                queryHeads,
                keyHeads
            )
            const weights = tf.add(contentWeights, positionWeights)
            attention = tf.softmax(weights, -1)
        } else {
            const combinedKeys = tf.concat([recentMemory.keys, keyHeads], 2)
            const combinedValues = tf.concat(
                [recentMemory.values, valueHeads],
                2
            )
            const relativePositionEncodings =
                this.relativePositionalEncoding(queryHeads)
            const positionWeights = tf.einsum(
                'bhqd,bhkd->bhqk',
                queryHeads,
                relativePositionEncodings
            )
            const contentWeights = tf.einsum(
                'bhqd,bhkd->bhqk',
                queryHeads,
                combinedKeys
            )
            const weights = tf.add(contentWeights, positionWeights)
            attention = tf.softmax(weights, -1)
        }

        attention = this.dropout.apply(attention)
        let attended
        if (recentMemory === null) {
            attended = tf.einsum('bhqk,bhkd->bhqd', attention, valueHeads)
        } else {
            attended = tf.einsum('bhqk,bhkd->bhqd', attention, combinedValues)
        }
        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, [batchSize, seqLength, this.units])

        const output = this.outputDense.apply(attended)
        const normalizedOutput = this.layerNormalization.apply(output)

        return normalizedOutput
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            units: this.units,
            heads: this.heads,
            dropout: this.dropout,
            epsilon: this.epsilon,
            memoryLength: this.memoryLength
        })
        return config
    }

    static get className() {
        return 'TransformerXLAttention'
    }
}

class Antirectifier extends LayerBase {
    constructor() {
        super({})
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
        this.invokeCallHook(inputs, kwargs)
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

    /**
     * If a custom layer class is to support serialization, it must implement
     * the `className` static getter.
     */
    static get className() {
        return 'Antirectifier'
    }
}

class RotaryPositionalEncoding extends LayerBase {
    constructor(config) {
        super({ ...config, name: `rot-${randomString()}` })
        this.units = config.units
        this.blockSize = config.blockSize
        this.posEncoding = null
    }

    build(inputShape) {
        const outputDim = inputShape[inputShape.length - 1]
        if (outputDim !== this.units) {
            throw new Error(
                `Embedding dimension mismatch. Expected ${this.units}, got ${outputDim}.`
            )
        }
        this.posEncoding = this.getRotaryPositionalEmbedding(
            this.blockSize,
            this.units
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
            const batchSize = inputs.shape[0]
            const seqLen = inputs.shape[1]
            const paddedInputs = inputs.pad([
                [0, 0],
                [0, Math.max(this.blockSize - seqLen, 0)],
                [0, 0]
            ])
            const paddedSeqLen = paddedInputs.shape[1]
            const posEncoding = this.posEncoding.slice(
                [0, 0],
                [paddedSeqLen, this.units]
            )
            const output = this.applyRotaryPositionalEmbedding(
                paddedInputs,
                posEncoding
            )
            return output.slice(
                [0, 0, 0],
                [batchSize, this.blockSize, this.units]
            )
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
        return [inputShape[0], this.blockSize, this.units]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            units: this.units,
            blockSize: this.blockSize
        })
        return config
    }

    static get className() {
        return 'RotaryPositionalEncoding'
    }
}

class CompressorHead extends LayerBase {
    constructor(config) {
        super({ ...config, name: `compressor-${randomString()}` })
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
            this.invokeCallHook(inputs, kwargs)
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

    static get className() {
        return 'CompressorHead'
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

    static get className() {
        return 'DumbCompression'
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

    static get className() {
        return 'ConvolutionalExpansionLayer'
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

    static get className() {
        return 'SequenceExpansionLayer'
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

    static get className() {
        return 'NearestNeighborUpsampling'
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

    static get className() {
        return 'LearnedUpsampling'
    }
}

// https://arxiv.org/abs/2203.03691
class HyperMixer extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units
    }

    build(inputShape) {
        // Assuming inputShape is [batchSize, numTokens, featureSize]
        const featureSize = inputShape[2]

        // Hypernetwork that generates weights for the main MLP dynamically
        // This can be a simple MLP itself or something more sophisticated
        this.dynamicWeightGenerator = tf.layers.dense({
            units: this.units * featureSize,
            activation: 'linear', // Or consider other activation functions
            useBias: true // Depending on whether you want biases in the weight generator
        })

        // Main MLP weights and bias, initialized here but will be generated dynamically
        // Note: These are placeholders; actual dynamic weights are generated per input
        this.kernel = tf.zeros([featureSize, this.units])
        this.bias = tf.zeros([this.units])

        super.build(inputShape) // Must call super.build() at the end
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Generate dynamic weights and biases from the input
            const weightBiasVector = this.dynamicWeightGenerator.apply(inputs)
            const dynamicWeights = weightBiasVector
                .slice([0, 0], [-1, this.units * inputs.shape[2]])
                .reshape([-1, inputs.shape[2], this.units])
            const dynamicBiases = weightBiasVector
                .slice([0, this.units * inputs.shape[2]], [-1, this.units])
                .reshape([-1, this.units])

            // Apply the dynamically generated weights and biases
            const output = tf.dot(inputs, dynamicWeights).add(dynamicBiases)
            return output
        })
    }

    computeOutputShape(inputShape) {
        // Output shape changes to [batchSize, numTokens, this.units]
        return [inputShape[0], inputShape[1], this.units]
    }

    getConfig() {
        const config = super.getConfig()
        config.units = this.units
        return config
    }

    static get className() {
        return 'HyperMixer'
    }
}

class StateSpace extends tf.layers.Layer {
    constructor(config) {
        super({ ...config, name: `ssm-${randomString()}` })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.epsilon = config.epsilon || 1e-5
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

        this.layernorm = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layernorm.build(inputShape)

        this._trainableWeights.push(...this.layernorm.trainableWeights)

        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
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
                const innerState = tf.tanh(
                    tf.add(
                        tf.add(
                            tf.matMul(input, kernel),
                            tf.matMul(state, recurrentKernel)
                        ),
                        bias
                    )
                )
                const newState = tf.matMul(innerState, outputKernel)
                outputs.push(newState)
                state = newState
            }

            let output = this.returnSequences
                ? tf.stack(outputs, 1)
                : outputs[outputs.length - 1]

            output = this.layernorm.apply(output)

            return this.residual.apply([inputs, output])
        })
    }

    computeOutputShape(inputShape) {
        const outputShape = this.returnSequences
            ? [inputShape[0], inputShape[1], this.units]
            : [inputShape[0], this.units]
        return outputShape
    }

    getConfig() {
        const config = {
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            epsilon: this.epsilon
        }
        const baseConfig = super.getConfig()
        Object.assign(config, baseConfig)
        return config
    }

    static get className() {
        return 'StateSpace'
    }
}

class FuzzyStateSpace extends tf.layers.Layer {
    constructor(config) {
        super({ ...config, name: `ssm-${randomString()}` })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.epsilon = config.epsilon || 1e-5
        this.chunkSize = config.chunkSize || 4
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

        this.layernorm = tf.layers.layerNormalization({
            epsilon: this.epsilon
        })
        this.layernorm.build(inputShape)

        this._trainableWeights.push(...this.layernorm.trainableWeights)

        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
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
                // state = tf.mean(
                //     newStateChunk.reshape([batchSize, chunkLength, this.units]),
                //     1
                // )
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

            output = this.layernorm.apply(output)

            return this.residual.apply([inputs, output])
        })
    }

    computeOutputShape(inputShape) {
        const outputShape = this.returnSequences
            ? [inputShape[0], inputShape[1], this.units]
            : [inputShape[0], this.units]
        return outputShape
    }

    getConfig() {
        const config = {
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            epsilon: this.epsilon
        }
        const baseConfig = super.getConfig()
        Object.assign(config, baseConfig)
        return config
    }

    static get className() {
        return 'FuzzyStateSpace'
    }
}

const exportedLayers = [
    Antirectifier,
    CausalSelfAttention,
    CompressorHead,
    ControlGate,
    ConvolutionalExpansionLayer,
    DebugLayer,
    DumbCompression,
    EncapsulatedMLP,
    LazyMixtureOfExperts,
    FuzzyStateSpace,
    GatedLinearUnit,
    GaussianMixtureModel,
    HyperMixer,
    LambdaLayer,
    LearnedUpsampling,
    MultiHeadAttention,
    MultiLayerPerceptron,
    NearestNeighborUpsampling,
    PassthroughLayer,
    Range,
    ResidualConnection,
    RotaryPositionalEncoding,
    SequenceExpansionLayer,
    SinusoidalPositionalEncoding,
    SparseMixtureOfExperts,
    StateSpace,
    SynthesizerAttention,
    TransformerXLAttention
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})
