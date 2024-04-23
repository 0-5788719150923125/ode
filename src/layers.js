import * as tf from '@tensorflow/tfjs'
import { GELU } from './activations.js'
import customOps from './ops.js'
import { randomString, seededPRNG, seededValueFromArray } from './utils.js'

const customLayers = {
    activation: (config) =>
        tf.layers.activation({ name: `act-${randomString()}`, ...config }),
    add: (config) =>
        tf.layers.add({ name: `add-${randomString()}`, ...config }),
    bottleneck: (config) =>
        tf.layers.dense({ name: `bot-${randomString()}`, ...config }),
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
    timeDistributed: (config) =>
        tf.layers.timeDistributed({ name: `time-${randomString()}`, ...config })
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
        return this.name
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

    call(inputs, kwargs, training) {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0]
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

class MultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super({ name: `mlp-${randomString()}`, ...config })
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || false
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
        this.inProjKernel = this.addWeight(
            'inProjKernel',
            [inputShape[inputShape.length - 1], this.innerDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.inProjBias = this.addWeight(
            'inProjBias',
            [this.innerDim],
            'float32',
            tf.initializers.zeros()
        )

        this.outProjKernel = this.addWeight(
            'outProjKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.outProjBias = this.addWeight(
            'outProjBias',
            [this.units],
            'float32',
            tf.initializers.zeros()
        )

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
            let outputs = this.dense(inputs, this.inProjKernel, this.inProjBias)
            if (this.customActivation) {
                outputs = this.customActivation.apply(outputs)
            } else if (this.activation !== 'linear') {
                outputs = tf.layers
                    .activation({ activation: this.activation })
                    .apply(outputs)
            }
            outputs = this.dense(outputs, this.outProjKernel, this.outProjBias)

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
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout
        }
    }
}

// class MultiLayerPerceptron extends LayerBase {
//     constructor(config) {
//         super({ name: `mlp-${randomString()}`, ...config })
//         this.units = config?.units || 256
//         this.innerDim = config?.innerDim || 1024
//         this.dropout = config?.dropout || 0
//         this.epsilon = config?.epsilon || false
//         if (config?.activation === 'gelu') {
//             this.customActivation = new GELU()
//             this.activation = 'linear'
//         } else {
//             this.activation = config?.activation || 'relu'
//         }
//         this.supportsMasking = true
//     }

//     build(inputShape) {
//         // Initialize dense layers for projection
//         this.inProj = tf.layers.dense({
//             units: this.innerDim,
//             inputDim: this.units,
//             activation: this.activation
//         })
//         this.outProj = tf.layers.dense({
//             units: this.units,
//             inputDim: inputShape,
//             activation: 'linear'
//         })

//         // Initialize layer normalization
//         if (this.epsilon) {
//             this.layernorm = tf.layers.layerNormalization({
//                 epsilon: this.epsilon
//             })
//         }

//         // Residual connections/skip connections are critical here
//         this.residual = new ResidualConnection()
//     }

//     call(inputs, kwargs, training = false) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             // Expand and contract projection via feedfoward layers
//             let outputs = this.inProj.apply(inputs)
//             if (this.customActivation) {
//                 outputs = this.customActivation.apply(outputs)
//             }
//             outputs = this.outProj.apply(outputs)
//             // If training, apply residual dropout
//             outputs = kwargs['training']
//                 ? tf.dropout(outputs, this.dropout)
//                 : outputs
//             // Apply layer norm
//             if (this.layernorm) outputs = this.layernorm.apply(outputs)
//             // Apply skip connection
//             return this.residual.apply([inputs, outputs])
//         })
//     }

//     computeOutputShape(inputShape) {
//         return inputShape
//     }

//     getConfig() {
//         return {
//             units: this.units,
//             innerDim: this.innerDim,
//             dropout: this.dropout
//         }
//     }
// }

class GatedLinearUnit extends MultiLayerPerceptron {
    constructor(config) {
        super({ name: `glu-${randomString()}`, ...config })
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

        // Initialize layer normalization
        if (this.epsilon) {
            this.layernorm = tf.layers.layerNormalization({
                epsilon: this.epsilon
            })
        }

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

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
            if (this.layernorm) outputs = this.layernorm.apply(outputs)
            // Apply skip connection
            return this.residual.apply([inputs, outputs])
        })
    }
}

class DenseMultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super({ name: `mlp-${randomString()}`, ...config })
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.epsilon = config?.epsilon || false
        this.activation = config?.activation || 'gelu'
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
        this.residual = new ResidualConnection()
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
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.bottleneck = config?.bottleneck || 128
        this.encoderActivation = config?.encoderActivation || 'relu'
        this.decoderActivation = config?.decoderActivation || 'relu'
        this.noise = config?.noise || 0
    }

    build(inputShape) {
        // Initialize dense layers for encoder
        this.encoder = tf.sequential()
        this.encoder.add(
            tf.layers.dense({
                units: this.innerDim,
                inputShape: inputShape,
                activation: this.encoderActivation
            })
        )
        this.encoder.add(
            tf.layers.dense({
                units: this.bottleneck,
                activation: 'linear'
            })
        )

        // Initialize dense layers for decoder
        this.decoder = tf.sequential()
        this.decoder.add(
            tf.layers.dense({
                units: this.innerDim,
                inputShape: [this.bottleneck],
                activation: this.decoderActivation
            })
        )
        this.decoder.add(
            tf.layers.dense({
                units: this.units,
                activation: 'linear'
            })
        )

        // Residual connections/skip connections are critical here
        this.residual = new ResidualConnection()
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            let processed = inputs

            if (this.noise > 0) {
                processed = processed.add(
                    tf.randomNormal(inputs.shape, 0, this.noise)
                )
            }

            // Encode the inputs to the bottleneck representation
            let outputs = this.encoder.apply(processed)
            // Decode the bottleneck representation back to the original dimensionality
            outputs = this.decoder.apply(outputs)
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
            bottleneck: this.bottleneck,
            encoderActivation: this.encoderActivation,
            decoderActivation: this.decoderActivation,
            noise: this.noise
        }
    }
}

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
            if (this.customActivation) {
                outputs = this.customActivation.apply(outputs)
            }
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

class SparseMixtureOfExperts extends LayerBase {
    constructor(config) {
        super({ name: `moe-${randomString()}`, ...config })
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

        // this._trainableWeights = [
        //     ...this.inGate.trainableWeights,
        //     ...this.hiddenGate.trainableWeights,
        //     ...this.outGate.trainableWeights
        // ]

        // Build each expert layer
        this.experts.forEach((expert) => {
            expert.build(inputShape)
            // this._trainableWeights.push(...expert.trainableWeights)
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

class SelfAttention extends LayerBase {
    constructor(config) {
        super({ name: `attn-${randomString()}`, ...config })
        this.units = config.units || 64
        this.projection = config.projection || 256
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

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf
                .matMul(Q, K, false, true)
                .div(tf.scalar(this.projection).sqrt())
                .add(mask)

            const weights = scores.softmax()

            const outputs = tf.matMul(weights, V)

            return this.residual.apply([inputs, outputs])
        })
    }

    applyDense(x, kernel) {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            projection: this.projection
        }
    }
}

// class SelfAttention extends LayerBase {
//     constructor(config) {
//         super({ name: `attn-${randomString()}`, ...config })
//         this.units = config.units || 64
//         this.projection = config.projection || 256
//     }

//     build(inputShape) {
//         this.query = tf.layers.dense({
//             units: this.projection,
//             activation: 'linear',
//             useBias: false,
//             kernelInitializer: tf.initializers.glorotUniform()
//         })
//         this.key = tf.layers.dense({
//             units: this.projection,
//             activation: 'linear',
//             useBias: false,
//             kernelInitializer: tf.initializers.glorotUniform()
//         })
//         this.value = tf.layers.dense({
//             units: this.units,
//             activation: 'linear',
//             useBias: false,
//             kernelInitializer: tf.initializers.glorotUniform()
//         })
//         this.residual = customLayers.ResidualConnection()
//     }

//     call(inputs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             const Q = this.query.apply(inputs)
//             const K = this.key.apply(inputs)
//             const V = this.value.apply(inputs)

//             const mask = tf.linalg
//                 .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
//                 .sub(tf.eye(inputs.shape[1]))
//                 .mul(tf.scalar(-1e9))

//             const scores = tf
//                 .matMul(Q, K, false, true)
//                 .div(tf.scalar(this.projection).sqrt())
//                 .add(mask)

//             const weights = scores.softmax()

//             const outputs = tf.matMul(weights, V)

//             return this.residual.apply([inputs, outputs])
//         })
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             units: this.units,
//             projection: this.projection
//         }
//     }
// }

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
        this.blockSize = config.blockSize
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const batchSize = inputs.shape[0]
            const seqLen = inputs.shape[1]
            const embeddingDim = inputs.shape[2]
            const paddedInputs = inputs.pad([
                [0, 0],
                [0, Math.max(this.blockSize - seqLen, 0)],
                [0, 0]
            ])
            const paddedSeqLen = paddedInputs.shape[1]
            const posEncoding = this.getRotaryPositionalEmbedding(
                paddedSeqLen,
                embeddingDim
            )
            const output = this.applyRotaryPositionalEmbedding(
                paddedInputs,
                posEncoding
            )
            return output.slice(
                [0, 0, 0],
                [batchSize, this.blockSize, embeddingDim]
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
        return [inputShape[0], this.blockSize, inputShape[2]]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            blockSize: this.blockSize
        }
    }
}

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
        this.epsilon = config.epsilon || false
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

            if (this.layernorm) output = this.layernorm.apply(output)

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
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            epsilon: this.epsilon
        }
    }
}

class ChunkedStateSpace extends LayerBase {
    constructor(config) {
        super({ name: `ssm-${randomString()}`, ...config })
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

        this.residual = new ResidualConnection()
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
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            epsilon: this.epsilon,
            chunkSize: this.chunkSize
        }
    }
}

class StructuredStateSpace extends LayerBase {
    constructor(config) {
        super({ name: `sss-${randomString()}`, ...config })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.epsilon = config.epsilon || false
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

    computeOutputShape(inputShape) {
        const outputShape = this.returnSequences
            ? [inputShape[0], inputShape[1], this.units]
            : [inputShape[0], this.units]
        return outputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            epsilon: this.epsilon,
            chunkSize: this.chunkSize
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

class SharedEmbedding extends LayerBase {
    constructor(config) {
        super({ name: `emb-${randomString()}`, ...config })
        this.inputDim = config.inputDim
        this.outputDim = config.outputDim
        this.embeddingsInitializer =
            config.embeddingsInitializer || 'glorotUniform'
    }

    build(inputShape) {
        this.embeddings = this.addWeight(
            'embeddings',
            [this.inputDim, this.outputDim],
            'float32',
            tf.initializers[this.embeddingsInitializer]()
        )
        this.built = true
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
                return tf.reshape(embeddings, [
                    inputs.shape[0],
                    inputs.shape[1],
                    this.outputDim
                ])
            } else if (inputs.shape.length === 3) {
                // Output projection
                const denseOutput = tf.matMul(
                    tf.reshape(inputs, [-1, this.outputDim]),
                    this.embeddings.read(),
                    false,
                    true
                )
                return tf.reshape(denseOutput, [
                    inputs.shape[0],
                    inputs.shape[1],
                    this.inputDim
                ])
            } else {
                throw new Error(
                    'Invalid input shape for SharedEmbedding layer.'
                )
            }
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            inputDim: this.inputDim,
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

const exportedLayers = [
    Antirectifier,
    Bias,
    CapsNet,
    CausalSelfAttention,
    ChunkedStateSpace,
    CollapseOneHot,
    CompressorHead,
    ControlGate,
    ConvolutionalExpansionLayer,
    DebugLayer,
    DepthwiseSeparableConvolution,
    DeterministicEmbedding,
    Autoencoder,
    DimensionContraction,
    DimensionExpansion,
    DumbCompression,
    EfficientChannelAttention,
    FourierFeaturePositionalEncoding,
    InstanceNormalization,
    GatedLinearUnit,
    LambdaLayer,
    LazyMixtureOfExperts,
    LearnedUpsampling,
    MixtureOfDepthsRouting,
    MultiLayerPerceptron,
    DenseMultiLayerPerceptron,
    NearestNeighborUpsampling,
    PerformerAttention,
    PseudoQuantumState,
    QuantumStateMachine,
    Range,
    ResidualConnection,
    RotaryPositionalEncoding,
    SequenceExpansionLayer,
    SharedEmbedding,
    SelfAttention,
    SinusoidalPositionalEncoding,
    SparseMixtureOfExperts,
    SqueezeAndExcitation,
    StateSpace,
    StructuredStateSpace,
    SynthesizerAttention,
    TemporalPooling,
    ToOneHot,
    Interrogator,
    Collideoscope,
    SparseEstimatedAttention,
    LinearAttention
]

exportedLayers.forEach((LayerClass) => {
    tf.serialization.registerClass(LayerClass)
    const className = LayerClass.className || LayerClass.name
    customLayers[className] = (config) => new LayerClass(config)
})
