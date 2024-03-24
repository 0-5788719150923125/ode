import * as tf from '@tensorflow/tfjs'
import { GELU } from './activations.js'

export class DebugLayer extends tf.layers.Layer {
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

tf.serialization.registerClass(DebugLayer)

export class Range extends tf.layers.Layer {
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
tf.serialization.registerClass(Range)

export class CausalSelfAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.config = Object.assign({ name: 'attn' }, config)

        // Config
        this.blockSize = config.blockSize
        this.units = config.units
        this.numHeads = config.numHeads
        this.dropout = config.dropout
        this.bias = config.bias
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
        this.layerNorm = tf.layers.layerNormalization({ epsilon: 1e-5 })
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

            let x = this.layerNorm.apply(inputs)

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

            const cAttn = dense(x, this.cAttnKernel, this.cAttnBias)

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

            let y = tf.matMul(att, v)

            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, [B, T, C])
            y = dense(y, this.cProjKernel, this.cProjBias)
            y = kwargs['training'] ? tf.dropout(y, this.dropout) : y

            y = tf.layers.add().apply([inputs, x])

            return y
        })
    }

    static get className() {
        return 'CausalSelfAttention'
    }
}
tf.serialization.registerClass(CausalSelfAttention)

export class SinusoidalPositionalEncoding extends tf.layers.Layer {
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

tf.serialization.registerClass(SinusoidalPositionalEncoding)

export class MultiHeadAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads
        this.units = config.units
        this.depth = this.units / this.numHeads
        this.supportsMasking = true
    }

    build(inputShape) {
        // Create dense layers for queries, keys, values and output
        const layers = ['query', 'key', 'value', 'out']
        layers.map((type, i) => {
            this[type] = tf.layers.dense({
                units: this.units,
                kernelInitializer: 'glorotUniform'
            })
            this[type].build(inputShape)
            this._trainableWeights.push(...this[type].trainableWeights)
        })
        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()
        // Initialize layer normalizations
        this.layernorm = tf.layers.layerNormalization({ epsilon: 1e-5 })
        this.layernorm.build(inputShape)
        this._trainableWeights.push(...this.layernorm.trainableWeights)

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)

            const batchSize = inputs.shape[0]
            const seqLength = inputs.shape[1]

            let q = this.query.apply(inputs)
            let k = this.key.apply(inputs)
            let v = this.value.apply(inputs)

            // Split the heads for multi-head attention
            q = tf
                .reshape(q, [batchSize, seqLength, this.numHeads, this.depth])
                .transpose([0, 2, 1, 3])
            k = tf
                .reshape(k, [batchSize, seqLength, this.numHeads, this.depth])
                .transpose([0, 2, 3, 1])
            v = tf
                .reshape(v, [batchSize, seqLength, this.numHeads, this.depth])
                .transpose([0, 2, 1, 3])

            // Compute raw attention scores
            let attentionScores = tf
                .matMul(q, k)
                .div(tf.sqrt(tf.scalar(this.depth)))

            if (kwargs?.mask) {
                // Invert the mask: true for tokens (no padding) becomes false, false for padding becomes true
                const maskInverted = kwargs.mask.logicalNot()
                // Convert the inverted mask to float and apply penalty to positions now marked as true (previously padding)
                const maskPenalty = tf.cast(maskInverted, 'float32').mul(-1e9)
                // Expand dimensions to make the mask compatible with attention scores
                const maskExpanded = maskPenalty.expandDims(1).expandDims(2)
                // Apply the expanded mask to the attention scores
                attentionScores = tf.add(attentionScores, maskExpanded)
            }

            const attentionWeights = tf.softmax(attentionScores, -1) // Apply softmax with mask applied
            let attentionOutput = tf
                .matMul(attentionWeights, v)
                .transpose([0, 2, 1, 3])
                .reshape([batchSize, seqLength, this.units])

            // Calculate attention scores
            attentionOutput = this.out.apply(attentionOutput)
            // Apply Residual Connection around Multi-Head Attention
            attentionOutput = this.residual.apply([inputs, attentionOutput])
            // Apply Layer Normalization
            attentionOutput = this.layernorm.apply(attentionOutput)

            return attentionOutput
        })
    }

    getConfig() {
        return {
            numHeads: this.numHeads,
            units: this.units
        }
    }

    static get className() {
        return 'MultiHeadAttention'
    }
}

tf.serialization.registerClass(MultiHeadAttention)

export class MultiLayerPerceptron extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.units = config?.units || 256
        this.numHeads = config?.numHeads || 8
        this.innerDim = config?.innerDim || 1024
        if (config?.activation === 'gelu') {
            this.customActivation = new GELU()
            this.activation = 'linear'
        } else {
            this.activation = config?.activation || 'relu'
        }
        this.dropout = config?.dropout || 0
        this.supportsMasking = true
    }

    build(inputShape) {
        // Initialize dense layers for projection
        this.in_proj = tf.layers.dense({
            units: this.innerDim,
            inputDim: this.units,
            activation: this.activation
            // inputShape: [null, null, null]
        })
        this.out_proj = tf.layers.dense({
            units: this.units,
            inputDim: this.innerDim
            // inputShape: [null, null, null]
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
            // Apply layer norm to inputs
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
            numHeads: this.numHeads,
            innerDim: this.innerDim
        }
    }

    static get className() {
        return 'MultiLayerPerceptron'
    }
}

tf.serialization.registerClass(MultiLayerPerceptron)

export class ResidualConnection extends tf.layers.Layer {
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

tf.serialization.registerClass(ResidualConnection)

// Originally adapted from:
// https://gist.githubusercontent.com/BenjaminWegener/311292080a71becbe5a8c0cc7657657d/raw/fd4f1f96184b58dace1854d0440d8c9dde3fd712/attention_layer_tfjs
// export class CausalAttentionLayer extends tf.layers.Layer {
//     constructor(config) {
//         super(config)
//         this.units = config.units || 256
//     }

//     build(inputShape) {
//         // Initialize the necessary dense layers for internal transformations
//         const layers = ['query', 'key', 'value']
//         layers.map((type, i) => {
//             this[type] = tf.layers.dense({
//                 units: this.units,
//                 kernelInitializer: 'glorotUniform'
//             })
//             this[type].build(inputShape)
//             this._trainableWeights.push(...this[type].trainableWeights)
//         })

//         super.build(inputShape)
//     }

//     computeOutputShape(inputShape) {
//         return inputShape
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs
//             this.invokeCallHook(inputs, kwargs)

//             const queries = this.query.apply(inputs)
//             const keys = this.key.apply(inputs).transpose([0, 2, 1])
//             const values = this.value.apply(inputs)

//             let scores = tf.matMul(queries, keys)
//             scores = tf.div(scores, tf.sqrt(tf.scalar(this.units)))

//             // // Integrate the padding mask if provided
//             if (kwargs?.mask) {
//                 // Flip the mask: True for non-padding becomes False, False for padding becomes True
//                 const maskInverted = kwargs.mask.logicalNot()
//                 // Convert the inverted mask to float and apply a penalty to positions now marked as True (previously padding)
//                 const maskPenalty = maskInverted.cast('float32').mul(-1e9)
//                 // Expand dimensions to make the mask compatible with attention scores
//                 const maskExpanded = maskPenalty.expandDims(1)
//                 // Apply the expanded mask to the attention scores
//                 scores = tf.add(scores, maskExpanded)
//             }

//             // Manually creating a causal mask
//             const seqLen = queries.shape[1]
//             const onesUpperTriangle = tf
//                 .ones([seqLen, seqLen])
//                 .cumsum(0)
//                 .cumsum(1)
//                 .greaterEqual(1)
//             const mask = onesUpperTriangle
//                 .logicalNot()
//                 .cast('float32')
//                 .mul(-1e9)
//             const maskExpanded = mask
//                 .expandDims(0)
//                 .tile([queries.shape[0], 1, 1])

//             scores = tf.add(scores, maskExpanded)

//             // compute the scaled dot product
//             const attentionWeights = tf.softmax(scores, -1)
//             const contextVector = tf.matMul(attentionWeights, values)

//             return contextVector
//         })
//     }

//     static get className() {
//         return 'CausalAttentionLayer'
//     }
// }

// tf.serialization.registerClass(CausalAttentionLayer)

export class ExpandDims extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.axis = config?.axis || -1
        this.supportsMasking = true
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)
            return tf.expandDims(inputs, this.axis)
        })
    }

    computeOutputShape(inputShape) {
        // This modifies the input shape by adding a 1 in the specified axis position.
        let outputShape = inputShape.slice()
        if (this.axis >= 0) {
            outputShape.splice(this.axis, 0, 1)
        } else {
            // When axis is -1, the new axis is added at the end of the shape.
            outputShape.push(1)
        }
        return outputShape
    }

    static get className() {
        return 'ExpandDims'
    }
}

tf.serialization.registerClass(ExpandDims)

export class LearnedPositionalEmbeddings extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.vocabSize = config.vocabSize
        this.maxSeqLength = config.maxSeqLength
        this.units = config.units
    }

    build(inputShape) {
        // Since this is a layer initialization method, weights are typically initialized here
        this.tokenEmbeddings = this.addWeight(
            'token_embeddings',
            [this.vocabSize, this.units],
            null,
            tf.initializers.glorotUniform()
        )
        this.positionalEmbeddings = this.addWeight(
            'positional_embeddings',
            [this.maxSeqLength, this.units],
            null,
            tf.initializers.glorotUniform()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            // Ensure inputs is not an array and is cast to int32 if it's used as indices
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            this.invokeCallHook(inputs, kwargs)

            // Ensure token indices are integers
            inputs = tf.cast(inputs, 'int32')

            const tokenIndices = inputs
            const batchSize = inputs.shape[0]
            const sequenceLength = inputs.shape[1]

            // Gather token embeddings, ensuring indices are int32
            const tokenEmbeddings = tf.gather(
                this.tokenEmbeddings.read(),
                tokenIndices.flatten()
            )

            // The reshape here assumes the flattening and gathering does not change the overall expected shape
            const reshapedTokenEmbeddings = tokenEmbeddings.reshape([
                batchSize,
                sequenceLength,
                this.units
            ])

            // Create a range tensor for positional indices and ensure it's int32
            const positions = tf.range(0, sequenceLength, 1, 'int32')
            const positionalEmbeddings = tf.gather(
                this.positionalEmbeddings.read(),
                positions
            )

            // Expanding and tiling the positional embeddings to match the batch size
            const reshapedPositionalEmbeddings = positionalEmbeddings
                .expandDims(0)
                .tile([batchSize, 1, 1])

            // Combine the embeddings
            return tf.add(reshapedTokenEmbeddings, reshapedPositionalEmbeddings)
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            vocabSize: this.vocabSize,
            maxSeqLength: this.maxSeqLength,
            units: this.units
        })
        return config
    }

    static get className() {
        return 'LearnedPositionalEmbeddings'
    }
}

tf.serialization.registerClass(LearnedPositionalEmbeddings)

export class LastTokenSelectionLayer extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.supportsMasking = true
    }

    computeOutputShape(inputShape) {
        // Transform input shape of [batchSize, sequenceLength, vocabSize]
        // to an output shape of [batchSize, vocabSize], removing
        // all predicted tokens except for the last
        return [inputShape[0], inputShape[2]]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const sequenceLength = inputs.shape[1]
            const lastTokenIndex = sequenceLength - 1

            const lastTokenSlice = inputs
                .slice([0, lastTokenIndex, 0], [-1, 1, -1])
                .squeeze([1])

            return lastTokenSlice
        })
    }

    static get className() {
        return 'LastTokenSelectionLayer'
    }
}

tf.serialization.registerClass(LastTokenSelectionLayer)

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
