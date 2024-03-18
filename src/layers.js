import * as tf from '@tensorflow/tfjs'

export class SinusoidalPositionalEncoding extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.supportsMasking = true
        this.maxSeqLength = config.maxSeqLength || 64
        this.embeddingDim = config.embeddingDim || 256
        // Pre-compute the positional encoding matrix for the maximum sequence length.
        this.posEncoding = this.precomputePositionalEncoding(
            this.maxSeqLength,
            this.embeddingDim
        )
        this.testPositionalEncoding()
    }

    precomputePositionalEncoding(seqLength, embeddingDim) {
        return tf.tidy(() => {
            const pos = tf.range(0, seqLength, 1, 'float32').expandDims(1)
            const i = tf.range(0, embeddingDim / 2, 1, 'float32').expandDims(0) // Only need half as many i values
            const angleRates = tf.pow(10000, tf.div(tf.mul(i, 2), embeddingDim)) // Multiply i by 2 here

            const angles = pos.div(angleRates)

            const sines = angles.sin()
            const cosines = angles.cos()

            // Stack sines and cosines in depth (along last dimension) and then interleave them
            const stacked = tf.stack([sines, cosines], 2) // Shape [seqLength, embeddingDim/2, 2]
            const posEncoding = stacked.reshape([seqLength, embeddingDim])
            return posEncoding.expandDims(0) // Add batch dimension
        })
    }

    testPositionalEncoding() {
        tf.tidy(() => {
            const seqLength = 4
            const embeddingDim = 8

            const posEncodingTensor = this.precomputePositionalEncoding(
                seqLength,
                embeddingDim
            )

            posEncodingTensor.print()
        })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Dynamically adjust the positional encoding to match the input shape
            const inputSeqLength = inputs.shape[1]
            const batchSize = inputs.shape[0]

            // Use slicing to adjust the positional encoding to the current input sequence length
            const posEncodingSliced = tf.slice(
                this.posEncoding,
                [0, 0, 0],
                [1, inputSeqLength, -1]
            )

            // Ensure positional encoding is broadcasted correctly over the batch size
            const posEncodingTiled = tf.tile(posEncodingSliced, [
                batchSize,
                1,
                1
            ])

            return tf.add(inputs, posEncodingTiled)
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    static get className() {
        return 'SinusoidalPositionalEncoding'
    }
}

tf.serialization.registerClass(SinusoidalPositionalEncoding)

class MultiHeadAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.supportsMasking = true
        this.numHeads = config.numHeads
        this.units = config.units
        this.depth = this.units / this.numHeads
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

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs

        let batchSize = inputs.shape[0]
        let seqLength = inputs.shape[1]

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

        let attentionWeights = tf.softmax(attentionScores, -1) // Apply softmax with mask applied
        let attentionOutput = tf
            .matMul(attentionWeights, v)
            .transpose([0, 2, 1, 3])
        attentionOutput = tf.reshape(attentionOutput, [
            batchSize,
            seqLength,
            this.units
        ])

        return this.out.apply(attentionOutput)
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

export class TransformerBlock extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.supportsMasking = true
        this.units = config?.units || 256
        this.numHeads = config?.numHeads || 8
        this.innerDim = config?.innerDim || 1024
        this.activation = config?.activation || 'relu'
    }

    build(inputShape) {
        // Initialize MultiHeadAttention once per block
        this.multiHeadAttention = new MultiHeadAttention({
            numHeads: this.numHeads,
            units: this.units
        })
        this.multiHeadAttention.build(inputShape)

        // Initialize dense layers for projection
        this.largeFeedforward = tf.layers.dense({
            units: this.innerDim,
            activation: this.activation
        })
        this.smallFeedforward = tf.layers.dense({ units: this.units })

        // Manually call build on dense layers to initialize weights
        this.largeFeedforward.build(inputShape)
        this.smallFeedforward.build([
            inputShape[0],
            inputShape[1],
            this.innerDim
        ])

        // Residual connections/skip connections are critical here
        this.attentionResidualConnection = new ResidualConnection()
        this.ffnResidualConnection = new ResidualConnection()

        // Initialize layer normalizations
        this.layernorm1 = tf.layers.layerNormalization({ epsilon: 1e-6 })
        this.layernorm2 = tf.layers.layerNormalization({ epsilon: 1e-6 })
        this.layernorm1.build(inputShape)
        this.layernorm2.build(inputShape)

        // Collect all trainable weights from internal layers
        this._trainableWeights = [
            ...this.multiHeadAttention.trainableWeights,
            ...this.largeFeedforward.trainableWeights,
            ...this.smallFeedforward.trainableWeights,
            ...this.layernorm1.trainableWeights,
            ...this.layernorm2.trainableWeights
        ]

        super.build(inputShape)
    }

    call(inputs, kwargs, training = false) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        // Calculate attention scores
        let attnOutput = this.multiHeadAttention.apply(inputs, kwargs)
        // Apply Residual Connection around Multi-Head Attention
        attnOutput = this.attentionResidualConnection.apply([
            inputs,
            attnOutput
        ])
        // Apply Layer Normalization
        const normalized = this.layernorm1.apply(attnOutput)
        // Feed-Forward Network block
        let ffOutput = this.largeFeedforward.apply(normalized)
        ffOutput = this.smallFeedforward.apply(ffOutput)
        // Apply Residual Connection around Feed-Forward Network
        ffOutput = this.ffnResidualConnection.apply([normalized, ffOutput])
        // Apply layer norm before return
        return this.layernorm2.apply(ffOutput)
    }

    getConfig() {
        return {
            units: this.units,
            numHeads: this.numHeads,
            innerDim: this.innerDim
        }
    }

    static get className() {
        return 'TransformerBlock'
    }
}

tf.serialization.registerClass(TransformerBlock)

export class ResidualConnection extends tf.layers.Layer {
    constructor() {
        super()
    }

    computeOutputShape(inputShape) {
        // inputShape[0] and inputShape[1 should be identical
        return inputShape[0]
    }

    call(inputs) {
        // inputs is an array where inputs[0] is the original input and inputs[1] is the output to be added to it.
        if (inputs.length !== 2) {
            throw new Error('ResidualConnection expects 2 inputs.')
        }

        const [originalInput, blockOutput] = inputs
        return tf.add(originalInput, blockOutput)
    }

    static get className() {
        return 'ResidualConnection'
    }
}

tf.serialization.registerClass(ResidualConnection)

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

// Originally adapted from:
// https://gist.githubusercontent.com/BenjaminWegener/311292080a71becbe5a8c0cc7657657d/raw/fd4f1f96184b58dace1854d0440d8c9dde3fd712/attention_layer_tfjs
export class CausalAttentionLayer extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.units = config.units || 256
        this.kernelInitializer = config.kernelInitializer || 'glorotUniform'
    }

    build(inputShape) {
        // Initialize the necessary dense layers for internal transformations
        this.queryDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: this.kernelInitializer
        })
        this.keyDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: this.kernelInitializer
        })
        this.valueDense = tf.layers.dense({
            units: this.units,
            kernelInitializer: this.kernelInitializer
        })

        // Ensuring internal layers are ready to be built with proper input shape
        const lastDimension = inputShape[inputShape.length - 1]
        this.queryDense.build([null, lastDimension])
        this.keyDense.build([null, lastDimension])
        this.valueDense.build([null, lastDimension])

        // REQUIRED: collecting weights from the internal layers manually
        this._trainableWeights = [
            ...this.queryDense.trainableWeights,
            ...this.keyDense.trainableWeights,
            ...this.valueDense.trainableWeights
        ]

        super.build(inputShape) // Mark the layer as built
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs) {
        return tf.tidy(() => {
            const queries = this.queryDense.apply(inputs)
            const keys = this.keyDense.apply(inputs)
            const values = this.valueDense.apply(inputs)

            const keysTransposed = tf.transpose(keys, [0, 2, 1])

            let scores = tf.matMul(queries, keysTransposed)
            scores = tf.div(scores, tf.sqrt(tf.scalar(this.units)))

            // Manually creating a causal mask
            const seqLen = queries.shape[1]
            const onesUpperTriangle = tf
                .ones([seqLen, seqLen])
                .cumsum(0)
                .cumsum(1)
                .greaterEqual(1)
            const mask = onesUpperTriangle
                .logicalNot()
                .cast('float32')
                .mul(-1e9)
            const maskExpanded = mask
                .expandDims(0)
                .tile([queries.shape[0], 1, 1])

            scores = tf.add(scores, maskExpanded)

            // compute the scaled dot product
            const attentionWeights = tf.softmax(scores, -1)
            const contextVector = tf.matMul(attentionWeights, values)

            return contextVector
        })
    }

    static get className() {
        return 'CausalAttentionLayer'
    }
}

tf.serialization.registerClass(CausalAttentionLayer)

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
