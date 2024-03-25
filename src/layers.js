import * as tf from '@tensorflow/tfjs'
import { GELU } from './activations.js'

const customLayers = {}
export default customLayers

class DebugLayer extends tf.layers.Layer {
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

class Range extends tf.layers.Layer {
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

class CausalSelfAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.config = Object.assign({ name: 'attn' }, config)

        // Config
        this.blockSize = config.blockSize || 256
        this.units = config.units || 256
        this.numHeads = config.numHeads || 4
        this.dropout = config.dropout || 0
        this.bias = config.bias || false
        this.epsilon = config.epsilon || 1e-5
        // Causal mask
        this.mask = tf.linalg.bandPart(
            tf.ones([config.blockSize, config.blockSize]),
            -1,
            0
        )
    }

    build(inputShape) {
        this.cAttnKernel = this.addWeight(
            'c_attn/kernel',
            [this.units, 3 * this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.cAttnBias = this.addWeight(
            'c_attn/bias',
            [3 * this.units],
            'float32',
            tf.initializers.zeros()
        )
        this.cProjKernel = this.addWeight(
            'c_proj/kernel',
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.cProjBias = this.addWeight(
            'c_proj/bias',
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

            let outputs = this.layerNorm.apply(inputs)

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

            const cAttn = dense(outputs, this.cAttnKernel, this.cAttnBias)

            // Make order of qkv split to follow minGPT
            let [q, k, v] = tf.split(cAttn, 3, -1)
            const [B, T, C] = k.shape

            const splitHeads = (x) =>
                tf.transpose(
                    tf.reshape(x, [B, T, this.numHeads, C / this.numHeads]),
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

            outputs = tf.matMul(att, v)

            outputs = tf.transpose(outputs, [0, 2, 1, 3])
            outputs = tf.reshape(outputs, [B, T, C])
            outputs = dense(outputs, this.cProjKernel, this.cProjBias)
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return tf.layers.add().apply([inputs, outputs])
        })
    }

    static get className() {
        return 'CausalSelfAttention'
    }
}

class SinusoidalPositionalEncoding extends tf.layers.Layer {
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

class MultiHeadAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads
        this.units = config.units
        this.depth = this.units / this.numHeads
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

        this.layerNorm = tf.layers.layerNormalization({ epsilon: 1e-5 })
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
                .reshape(q, [batchSize, -1, this.numHeads, this.depth])
                .transpose([0, 2, 1, 3])
            k = tf
                .reshape(k, [batchSize, -1, this.numHeads, this.depth])
                .transpose([0, 2, 3, 1])
            v = tf
                .reshape(v, [batchSize, -1, this.numHeads, this.depth])
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
            numHeads: this.numHeads,
            units: this.units,
            useCausalMask: this.useCausalMask
        }
    }

    static get className() {
        return 'MultiHeadAttention'
    }
}

class MultiLayerPerceptron extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config.epsilon || 1e-5
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
        this.in_proj = tf.layers.dense({
            units: this.innerDim,
            inputDim: this.units,
            activation: this.activation
        })
        this.out_proj = tf.layers.dense({
            units: this.units,
            inputDim: this.innerDim
        })

        // Manually call build on dense layers to initialize weights
        this.in_proj.build(inputShape)
        this.out_proj.build([inputShape[1], this.innerDim])

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()

        // Initialize layer normalization
        this.layernorm = tf.layers.layerNormalization({ epsilon: 1e-5 })
        this.layernorm.build(inputShape)

        // Collect all trainable weights from internal layers
        this._trainableWeights = [
            ...this.in_proj.trainableWeights,
            ...this.out_proj.trainableWeights,
            ...this.layernorm.trainableWeights
        ]

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
            // Apply layer norm
            let outputs = this.layernorm.apply(inputs)
            // Expand and contract projection via feedfoward layers
            outputs = this.in_proj.apply(outputs)
            if (this.customActivation) {
                outputs = this.customActivation.apply(outputs)
            }
            outputs = this.out_proj.apply(outputs)
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

class ResidualConnection extends tf.layers.Layer {
    constructor() {
        super()
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

class GaussianMixtureModel extends tf.layers.Layer {
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

class GatedLinearUnit extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.supportsMasking = true
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.linearKernel = this.addWeight(
            'linearKernel',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform({})
        )
        this.gateKernel = this.addWeight(
            'gateKernel',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform({})
        )
        this.linearBias = this.addWeight(
            'linearBias',
            [inputDim],
            'float32',
            tf.initializers.zeros()
        )
        this.gateBias = this.addWeight(
            'gateBias',
            [inputDim],
            'float32',
            tf.initializers.zeros()
        )
        super.build(inputShape)
    }

    call(inputs, kwargs) {
        let input = inputs
        if (Array.isArray(input)) {
            input = input[0]
        }

        this.invokeCallHook(inputs, kwargs)

        // Use tf.tidy for better memory management
        return tf.tidy(() => {
            const linearPart = tf
                .matMul(input, this.linearKernel.read())
                .add(this.linearBias.read())

            const gatePart = tf
                .matMul(input, this.gateKernel.read())
                .add(this.gateBias.read())
                .sigmoid()

            return linearPart.mul(gatePart)
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        const baseConfig = super.getConfig()
        return {
            ...baseConfig
        }
    }

    static get className() {
        return 'GatedLinearUnit'
    }
}

// Originally adapted from:
// https://gist.githubusercontent.com/BenjaminWegener/311292080a71becbe5a8c0cc7657657d/raw/fd4f1f96184b58dace1854d0440d8c9dde3fd712/attention_layer_tfjs
class LambdaLayer extends tf.layers.Layer {
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

// https://github.com/iafarhan/causal-synthesizer-multihead-attention/blob/main/synthesizer.py
class SynthesizerAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.units = config.units // Embedding size (n_embd)
        this.numHeads = config.numHeads // Number of attention heads (n_head)
        this.blockSize = config.blockSize // Sequence length (block_size)
        this.attnDropoutRate = config.dropout // Attention dropout rate (attn_pdrop)
        this.residDropoutRate = config.dropout // Residual dropout rate (resid_pdrop)
        this.d_k = this.units / this.numHeads // Dimensionality of each head

        // Initialize layers and variables
        this.w1 = tf.layers.dense({
            units: this.units,
            activation: 'relu',
            useBias: true
        })
        this.w2 = null // Initialized in build to use dynamic shapes
        this.b2 = null // Initialized in build
        this.value = tf.layers.dense({ units: this.units, useBias: false })
        this.proj = tf.layers.dense({ units: this.units })
        this.attnDrop = tf.layers.dropout({ rate: this.attnDropoutRate })
        this.residDrop = tf.layers.dropout({ rate: this.residDropoutRate })
    }

    build(inputShape) {
        // Dynamic weight initialization
        this.w2 = this.addWeight(
            'w2',
            [this.d_k, this.blockSize - 1],
            'float32',
            tf.initializers.randomUniform({ minVal: -0.001, maxVal: 0.001 })
        )
        this.b2 = this.addWeight(
            'b2',
            [this.blockSize - 1],
            'float32',
            tf.initializers.zeros()
        )
        super.build(inputShape) // Call the super method at the end
    }

    call(inputs, { training = false } = {}) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs // Handle both single inputs and arrays

        const x = inputs
        const [B, T, _] = x.shape

        // Step 1: Apply w1 with ReLU activation and prepare for multi-head attention
        let reluOut = this.w1
            .apply(x)
            .reshape([B, T, this.numHeads, this.d_k])
            .transpose([0, 2, 1, 3])

        // Step 2: Value projection
        let v = this.value
            .apply(x)
            .reshape([B, T, this.numHeads, this.d_k])
            .transpose([0, 2, 1, 3])

        // Step 3: Compute scores with synthesizer mechanism
        let scores = tf.matMul(reluOut, this.w2.read()).add(this.b2.read())

        // Apply causal mask (if necessary, based on your application's needs)
        // Note: Skipping mask application in this code for simplicity as it's not detailed in your request

        // Step 4: Apply softmax to scores
        let probAttn = tf.softmax(scores, -1)

        // Step 5: Apply attention to values and reshape
        // Assuming probAttn has shape [1, 8, 256, 255] and we want to multiply it by v
        // First, ensure v is prepared with the correct shape. It should be compatible for matMul operation
        // Let's say v needs to be [1, 8, 255, units/numHeads] for the multiplication to be valid

        // Assuming the reshaping and preparation of v has been correctly done before this step:
        let y = tf.matMul(probAttn, v) // Now, this should work as expected, given v is correctly shaped

        // After matMul, reshape y to match the expected output dimensions
        y = y.transpose([0, 2, 1, 3]).reshape([B, T, this.units])

        // Step 6: Apply final projection and dropout (if training)
        y = this.proj.apply(y)
        if (training) {
            y = this.residDrop.apply(y)
        }

        return y
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            numHeads: this.numHeads,
            blockSize: this.blockSize,
            attnDropoutRate: this.attnDropoutRate,
            residDropoutRate: this.residDropoutRate
        }
    }

    static get className() {
        return 'SynthesizerAttention'
    }
}

const exportedLayers = [
    CausalSelfAttention,
    DebugLayer,
    GatedLinearUnit,
    GaussianMixtureModel,
    LambdaLayer,
    MultiHeadAttention,
    MultiLayerPerceptron,
    Range,
    ResidualConnection,
    SinusoidalPositionalEncoding,
    SynthesizerAttention
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})

// export class LearnedPositionalEmbeddings extends tf.layers.Layer {
//     constructor(config) {
//         super(config)
//         this.vocabSize = config.vocabSize
//         this.maxSeqLength = config.maxSeqLength
//         this.units = config.units
//     }

//     build(inputShape) {
//         // Since this is a layer initialization method, weights are typically initialized here
//         this.tokenEmbeddings = this.addWeight(
//             'token_embeddings',
//             [this.vocabSize, this.units],
//             null,
//             tf.initializers.glorotUniform()
//         )
//         this.positionalEmbeddings = this.addWeight(
//             'positional_embeddings',
//             [this.maxSeqLength, this.units],
//             null,
//             tf.initializers.glorotUniform()
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             // Ensure inputs is not an array and is cast to int32 if it's used as indices
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs
//             this.invokeCallHook(inputs, kwargs)

//             // Ensure token indices are integers
//             inputs = tf.cast(inputs, 'int32')

//             const tokenIndices = inputs
//             const batchSize = inputs.shape[0]
//             const sequenceLength = inputs.shape[1]

//             // Gather token embeddings, ensuring indices are int32
//             const tokenEmbeddings = tf.gather(
//                 this.tokenEmbeddings.read(),
//                 tokenIndices.flatten()
//             )

//             // The reshape here assumes the flattening and gathering does not change the overall expected shape
//             const reshapedTokenEmbeddings = tokenEmbeddings.reshape([
//                 batchSize,
//                 sequenceLength,
//                 this.units
//             ])

//             // Create a range tensor for positional indices and ensure it's int32
//             const positions = tf.range(0, sequenceLength, 1, 'int32')
//             const positionalEmbeddings = tf.gather(
//                 this.positionalEmbeddings.read(),
//                 positions
//             )

//             // Expanding and tiling the positional embeddings to match the batch size
//             const reshapedPositionalEmbeddings = positionalEmbeddings
//                 .expandDims(0)
//                 .tile([batchSize, 1, 1])

//             // Combine the embeddings
//             return tf.add(reshapedTokenEmbeddings, reshapedPositionalEmbeddings)
//         })
//     }

//     computeOutputShape(inputShape) {
//         return [inputShape[0], inputShape[1], this.units]
//     }

//     getConfig() {
//         const config = super.getConfig()
//         Object.assign(config, {
//             vocabSize: this.vocabSize,
//             maxSeqLength: this.maxSeqLength,
//             units: this.units
//         })
//         return config
//     }

//     static get className() {
//         return 'LearnedPositionalEmbeddings'
//     }
// }

// tf.serialization.registerClass(LearnedPositionalEmbeddings)

// class MixtureOfExpertsLayer extends tf.layers.Layer {
//     constructor(config) {
//         super(config)
//         this.expertCount = config.expertCount || 2 // Number of experts
//         this.units = config.units // Number of units in the gating and expert layers
//     }

//     build(inputShape) {
//         // Gating mechanism to decide which expert to use for each sample
//         this.gate = this.addWeight(
//             'gate',
//             [inputShape[inputShape.length - 1], this.expertCount],
//             'float32',
//             tf.initializers.glorotUniform({})
//         )

//         // Experts are simple Dense layers in this example
//         this.experts = []
//         for (let i = 0; i < this.expertCount; i++) {
//             this.experts.push(
//                 this.addWeight(
//                     `expert_${i}`,
//                     [inputShape[inputShape.length - 1], this.units],
//                     'float32',
//                     tf.initializers.glorotUniform({})
//                 )
//             )
//         }
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             const gateOutput = tf.softmax(tf.dot(inputs, this.gate.read()), 1) // Softmax to ensure output sums to 1
//             let output = null

//             for (let i = 0; i < this.expertCount; i++) {
//                 // Compute the output for each expert
//                 const expertOutput = tf.dot(inputs, this.experts[i].read())
//                 // Weight the output by the gating mechanism
//                 const weightedOutput = tf.mul(
//                     expertOutput,
//                     gateOutput.slice([0, i], [-1, 1])
//                 )

//                 if (output === null) {
//                     output = weightedOutput
//                 } else {
//                     output = tf.add(output, weightedOutput)
//                 }
//             }

//             return output
//         })
//     }

//     // TensorFlow.js requires the getConfig method to save and load models
//     getConfig() {
//         const config = super.getConfig()
//         config.expertCount = this.expertCount
//         config.units = this.units
//         return config
//     }

//     // Static method to help TensorFlow.js identify this class
//     static get className() {
//         return 'MixtureOfExpertsLayer'
//     }
// }

// tf.serialization.registerClass(MixtureOfExpertsLayer)

// class SimplifiedMoELayer extends tf.layers.Layer {
//     constructor(config) {
//         super(config)
//         this.units = config.units
//         this.expertCount = config.expertCount || 2
//     }

//     build(inputShape) {
//         // Initialize gating mechanism
//         this.gate = this.addWeight(
//             'gate',
//             [inputShape[inputShape.length - 1], this.expertCount],
//             'float32',
//             tf.initializers.glorotUniform({})
//         )

//         // Initialize experts
//         this.experts = []
//         for (let i = 0; i < this.expertCount; i++) {
//             let expert = tf.layers.dense({
//                 units: this.units,
//                 activation: 'relu', // Example activation
//                 kernelInitializer: 'glorotUniform',
//                 useBias: true
//             })
//             expert.build(inputShape) // Manually set input shape
//             this.experts.push(expert)
//         }
//     }

//     call(inputs) {
//         let gateScores = tf.matMul(inputs, this.gate.read())
//         gateScores = tf.softmax(gateScores) // Ensure scores sum to 1

//         // Example of simplifying by just using a weighted sum of experts
//         let output = null
//         for (let i = 0; i < this.expertCount; i++) {
//             let expertOutput = this.experts[i].apply(inputs)
//             let weightedOutput = tf.mul(
//                 expertOutput,
//                 gateScores.slice([0, i], [-1, 1])
//             )
//             if (output === null) {
//                 output = weightedOutput
//             } else {
//                 output = tf.add(output, weightedOutput)
//             }
//         }
//         return output
//     }

//     getConfig() {
//         return {
//             units: this.units,
//             expertCount: this.expertCount
//         }
//     }

//     static get className() {
//         return 'SimplifiedMoELayer'
//     }
// }

// tf.serialization.registerClass(SimplifiedMoELayer)

// // export class AttentionLayer extends tf.layers.Layer {
// //     constructor(config) {
// //         super(config)
// //         this.n = config.n
// //     }

// //     build(inputShape) {
// //         // Step 1: Define a learnable query vector (assuming your features size is `n`)
// //         this.query = tf.variable(tf.randomNormal([this.n, 1]))
// //     }

// //     call(inputs) {
// //         // Step 2: Compute attention scores using dot product
// //         // `lstmOutput` shape: [batch, timesteps, features]
// //         // `query` shape: [features, 1]
// //         // We need to perform a batch dot product, resulting in a shape of [batch, timesteps, 1] for scores
// //         const scores = tf.matMul(inputs, this.query, false, true)

// //         // Step 3: Apply softmax to get attention weights
// //         const weights = tf.softmax(scores, 1) // Softmax over the timesteps dimension

// //         // Step 4: Compute the context vector as a weighted sum of LSTM outputs
// //         // `weights` shape: [batch, timesteps, 1]
// //         // `lstmOutput` shape: [batch, timesteps, features]
// //         // We need to multiply and sum over the timesteps, resulting in [batch, features] for the context vector
// //         const output = tf.sum(tf.mul(inputs, weights), 1)

// //         return output
// //     }

// //     getConfig() {
// //         return {
// //             units: this.n
// //         }
// //     }

// //     static get className() {
// //         return 'AttentionLayer'
// //     }
// // }

// // class SparseMixtureOfExpertsLayer extends tf.layers.Layer {
// //     constructor(config) {
// //         super(config)
// //         this.expertCount = config.expertCount || 2 // Number of experts
// //         this.units = config.units // Number of units for each expert layer
// //     }

// //     build(inputShape) {
// //         // Initialize gating mechanism weights correctly
// //         // this.gate = this.addWeight(
// //         //     'gate',
// //         //     [inputShape[inputShape.length - 1], this.expertCount],
// //         //     'float32',
// //         //     tf.initializers.glorotUniform({})
// //         // )

// //         this.experts = []
// //         // Initialization of expert layers
// //         for (let i = 0; i < this.expertCount; i++) {
// //             const expertLayer = tf.layers.dense({
// //                 units: this.units, // This should match the model design, not necessarily inputShape / expertCount
// //                 kernelInitializer: 'glorotUniform',
// //                 useBias: true
// //             })
// //             // No need to call build here, as applying the layer will handle this
// //             this.experts.push(expertLayer)
// //         }
// //     }

// //     call(inputs, kwargs) {
// //         return tf.tidy(() => {
// //             // Given adjustments and explanation...
// //             const selectedExpertIndex = Math.floor(
// //                 Math.random() * this.expertCount
// //             )
// //             const expertLayer = this.experts[selectedExpertIndex]
// //             // Assuming inputs shape is [batchSize, 64] due to GRU output
// //             const output = expertLayer.apply(inputs) // Should work with [batchSize, 64] input shape
// //             return output
// //         })
// //     }

// //     // call(inputs) {
// //     //     console.error(inputs)
// //     //     return tf.tidy(() => {
// //     //         // Calculate gate scores
// //     //         const gateScores = tf.softmax(
// //     //             tf.matMul(inputs, this.gate.read()),
// //     //             -1
// //     //         ) // Ensure softmax for distribution

// //     //         let output = null
// //     //         // Apply each expert based on gate scores
// //     //         for (let i = 0; i < this.expertCount; i++) {
// //     //             const expertOutput = this.experts[i].apply(inputs)
// //     //             const weightedOutput = tf.mul(
// //     //                 expertOutput,
// //     //                 gateScores.slice([0, i], [-1, 1])
// //     //             )

// //     //             if (output === null) {
// //     //                 output = weightedOutput
// //     //             } else {
// //     //                 output = tf.add(output, weightedOutput)
// //     //             }
// //     //         }

// //     //         return output
// //     //     })
// //     // }

// //     getConfig() {
// //         const config = super.getConfig()
// //         config.expertCount = this.expertCount
// //         config.units = this.units
// //         return config
// //     }

// //     static get className() {
// //         return 'SparseMixtureOfExpertsLayer'
// //     }
// // }

// class SparseMixtureOfExpertsLayer extends tf.layers.Layer {
//     constructor(config) {
//         super(config)
//         this.expertCount = config.expertCount || 2 // Number of experts
//         this.units = config.units // Number of units for each expert layer
//     }

//     build(inputShape) {
//         this.experts = []
//         for (let i = 0; i < this.expertCount; i++) {
//             // Define each expert with the anticipated input shape
//             const expertLayer = tf.layers.dense({
//                 units: this.units,
//                 kernelInitializer: 'glorotUniform',
//                 useBias: true,
//                 inputShape: [this.inputDim * this.expertCount]
//             })
//             this.experts.push(expertLayer)
//         }
//         // This is required to set the layer's built property to true.
//         super.build(inputShape)
//     }

//     call(inputs, kwargs) {
//         this.invokeCallHook(inputs, kwargs)
//         console.log(model.summary())
//         return tf.tidy(() => {
//             const selectedExpertIndex = Math.floor(
//                 Math.random() * this.expertCount
//             )
//             const expertLayer = this.experts[selectedExpertIndex]
//             // Apply the selected expert layer to the inputs
//             const output = expertLayer.apply(inputs)
//             return output
//         })
//     }

//     getConfig() {
//         const config = super.getConfig()
//         config.expertCount = this.expertCount
//         config.units = this.units
//         return config
//     }

//     static get className() {
//         return 'SparseMixtureOfExpertsLayer'
//     }
// }

// tf.serialization.registerClass(SparseMixtureOfExpertsLayer)

// class SparseMoE {
//     constructor(options) {
//         this.numExperts = options.numExperts
//         this.expertLayers = []
//         this.gatingMechanism = tf.layers.dense({
//             units: this.numExperts,
//             activation: 'softmax'
//         })

//         // Initialize experts
//         for (let i = 0; i < this.numExperts; i++) {
//             const expertLayer = tf.layers.dense({
//                 units: options.expertOutputUnits
//             })
//             this.expertLayers.push(expertLayer)
//         }
//     }

//     apply(input) {
//         const gateOutputs = this.gatingMechanism.apply(input)
//         let output = null

//         // Assume a simple selection mechanism: choosing the expert with the highest gate output
//         const selectedIndex = tf.argMax(gateOutputs, 1)
//         output = tf
//             .oneHot(selectedIndex, this.numExperts)
//             .mul(gateOutputs)
//             .sum(1)

//         // Apply selected expert. This is a simplification, real implementation may differ.
//         const expertOutputs = this.expertLayers.map((expert, index) => {
//             // This step needs to dynamically select the expert based on the selectedIndex.
//             // A more complex implementation might be required for actual sparse selection.
//             return expert.apply(input).mul(output)
//         })

//         // Combine expert outputs. This simplistic approach assumes only one expert is active.
//         // In reality, you might combine outputs based on weights from the gating mechanism.
//         return tf.stack(expertOutputs).sum(0)
//     }

//     static get className() {
//         return 'SparseMoE'
//     }
// }

// tf.serialization.registerClass(SparseMoE)

// // Generate some mock data with corrected type casting for 'oneHot'
// const xTrain = tf.randomNormal([1000, 10, 16]) // 1000 samples, 10 time steps, 64 features per step
// const yIndices = tf.floor(tf.randomUniform([1000], 0, 10)).toInt() // Correctly cast to int32
// const yTrain = tf.oneHot(yIndices, 10) // 1000 labels, 10 classes

// // Define the model
// const model = tf.sequential()
// model.add(
//     tf.layers.gru({
//         units: 64,
//         returnSequences: false,
//         inputShape: [10, 16] // Ensure this matches your input data shape
//     })
// )
// // model.add(
// //     new SparseMoE({
// //         expertOutputUnits: 64,
// //         numExperts: 2
// //     })
// // )
// model.add(
//     new SparseMixtureOfExpertsLayer({
//         units: 64,
//         expertCount: 2,
//         inputDim: 64 // Pass the correct input dimension expected by experts
//     })
// )
// model.add(
//     tf.layers.dense({
//         units: 10,
//         activation: 'softmax'
//     })
// )

// model.compile({
//     optimizer: 'adam',
//     loss: 'categoricalCrossentropy',
//     metrics: ['accuracy']
// })

// console.log(model.summary())

// // Train the model
// model
//     .fit(xTrain, yTrain, {
//         epochs: 10,
//         batchSize: 32,
//         // callbacks: tf.callbacks.earlyStopping({ patience: 3 })
//         callbacks: {
//             onBatchEnd: (batch, logs) => {
//                 console.log(logs)
//             }
//         }
//     })
//     .then((info) => {
//         console.log('Training complete')
//         console.log('Final accuracy:', info.history.acc)
//     })
//     .catch((error) => {
//         console.error('Training failed', error)
//     })
