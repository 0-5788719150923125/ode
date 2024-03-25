import OmnipotentDiabolicalErudite from './model.v3.js'
import PretrainedTokenizer from './tokenizers.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends OmnipotentDiabolicalErudite
 */
import tf from '@tensorflow/tfjs'
export default class OriginalDecoderEngine extends OmnipotentDiabolicalErudite {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 256
        this.dropout = 0.1
        this.epsilon = 1e-5
    }

    trainTokenizer() {
        this.tokenizer = new PretrainedTokenizer()
    }

    build() {
        const inputs = tf.input({ shape: [null] })

        const tokEmb = tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform',
                embeddingsRegularizer: null,
                activityRegularizer: null
            })
            .apply(inputs)

        const range = new Range().apply(inputs)
        let posEmb = tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        let outputs
        outputs = tf.layers.add().apply([tokEmb, posEmb])
        outputs = tf.layers
            .dropout({
                name: 'drop',
                rate: this.dropout
            })
            .apply(outputs)

        // const inputs = this.tf.input({ shape: [null] })

        // const tokenEmbeddings = this.tf.layers
        //     .embedding({
        //         name: 'wte',
        //         inputDim: this.tokenizer.getLength(),
        //         outputDim: this.units,
        //         embeddingsInitializer: 'glorotUniform'
        //     })
        //     .apply(inputs)

        // // const range = this.ode.layers.Range().apply(inputs)
        // const range = new Range().apply(inputs)

        // const positionalEmbeddings = this.tf.layers
        //     .embedding({
        //         name: 'wpe',
        //         inputDim: this.config.contextLength,
        //         outputDim: this.units,
        //         embeddingsInitializer: 'glorotUniform'
        //     })
        //     .apply(range)

        // let outputs = this.tf.layers
        //     .add()
        //     .apply([tokenEmbeddings, positionalEmbeddings])

        // outputs = this.tf.layers
        //     .dropout({
        //         name: 'dropout',
        //         rate: this.dropout
        //     })
        //     .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = Block({
                name: 'gpt/h/' + i,
                blockSize: this.config.contextLength,
                nEmbd: this.units,
                nHead: 4,
                residDrop: this.dropout,
                dropout: this.dropout,
                bias: false,
                debug: false,
                tokEmb: true,
                lmHead: true
            }).apply(outputs)
            // outputs = this.ode.layers
            //     .CausalSelfAttention({
            //         blockSize: this.config.contextLength,
            //         units: this.units,
            //         numHeads: this.numHeads,
            //         dropout: this.dropout,
            //         epsilon: this.epsilon,
            //         bias: false
            //     })
            //     .apply(outputs)

            // outputs = this.ode.layers
            //     .MultiLayerPerceptron({
            //         units: this.units,
            //         innerDim: this.innerDim,
            //         numHeads: this.numHeads,
            //         dropout: this.dropout,
            //         epsilon: this.epsilon,
            //         activation: 'gelu'
            //     })
            //     .apply(outputs)
        }

        outputs = this.tf.layers
            .layerNormalization({
                name: 'head/ln',
                epsilon: this.epsilon
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                name: 'head',
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

class CausalSelfAttention extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.config = Object.assign({ name: 'attn' }, config)

        // Config
        this.blockSize = config.blockSize
        this.nEmbd = config.nEmbd
        this.nHead = config.nHead
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
            [this.nEmbd, 3 * this.nEmbd],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.cAttnBias = this.addWeight(
            'c_attn/bias',
            [3 * this.nEmbd],
            'float32',
            tf.initializers.zeros()
        )
        this.cProjKernel = this.addWeight(
            'c_proj/kernel',
            [this.nEmbd, this.nEmbd],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.cProjBias = this.addWeight(
            'c_proj/bias',
            [this.nEmbd],
            'float32',
            tf.initializers.zeros()
        )
    }

    computeOutputShape(inputShape) {
        // console.log('computeOutputShape', inputShape)
        return inputShape
        // return [null, this.blockSize, this.nEmbd]
    }

    getConfig() {
        // This is neeed to save and load the model
        // When the model is saved, the config is saved with it
        // When the model is loaded, the config is used to create a new instance of the layer
        const config = super.getConfig()
        return Object.assign({}, config, this.config)
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)

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

            const cAttn = dense(input, this.cAttnKernel, this.cAttnBias)

            // Make prder of qkv split to follow minGPT
            let [q, k, v] = tf.split(cAttn, 3, -1)
            const [B, T, C] = k.shape

            if (this.config.debug) {
                LogLayer({ name: 'att_x' }).call(input)
                LogLayer({ name: 'att_c_attn' }).call(cAttn)
                LogLayer({ name: 'att_q_before' }).call(q)
                LogLayer({ name: 'att_k_before' }).call(k)
                LogLayer({ name: 'att_v_before' }).call(v)
            }

            const splitHeads = (x) =>
                tf.transpose(
                    tf.reshape(x, [B, T, this.nHead, C / this.nHead]),
                    [0, 2, 1, 3]
                )

            q = splitHeads(q)
            k = splitHeads(k)
            v = splitHeads(v)

            if (this.config.debug) {
                LogLayer({ name: 'att_q_after' }).call(q)
                LogLayer({ name: 'att_k_after' }).call(k)
                LogLayer({ name: 'att_v_after' }).call(v)
            }

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
            if (this.config.debug) {
                LogLayer({ name: '> att_softmax' }).call(att)
            }

            let y = tf.matMul(att, v)
            if (this.config.debug) {
                LogLayer({ name: 'att_yv' }).call(y)
            }

            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, [B, T, C])
            y = dense(y, this.cProjKernel, this.cProjBias)
            y = kwargs['training'] ? tf.dropout(y, this.dropout) : y
            if (this.config.debug) {
                LogLayer({ name: 'att_y' }).call(y)
            }

            return y
        })
    }

    static get className() {
        return 'CausalSelfAttention'
    }
}
tf.serialization.registerClass(CausalSelfAttention)

class GELU extends tf.layers.Layer {
    constructor() {
        super({})
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            // In functional API, input is an array of tensors
            // So we need to get the first element (the actual input)
            // Add a check as here:
            // https://github.com/tensorflow/tfjs-examples/blob/master/custom-layer/custom_layer.js
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)
            const cdf = tf.mul(
                0.5,
                tf.add(
                    1,
                    tf.tanh(
                        tf.mul(
                            tf.sqrt(tf.div(2, Math.PI)),
                            tf.add(input, tf.mul(0.044715, tf.pow(input, 3)))
                        )
                    )
                )
            )
            return tf.mul(input, cdf)
        })
    }

    static get className() {
        return 'GELU'
    }
}
tf.serialization.registerClass(GELU)

function MLP(conf) {
    const config = Object.assign({ name: 'mlp' }, conf)
    const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] })
    let x
    x = tf.layers
        .dense({
            name: config.name + '/c_fc',
            units: 4 * config.nEmbd,
            inputDim: config.nEmbd,
            inputShape: [config.blockSize, config.nEmbd]
        })
        .apply(inputs)
    x = new GELU().apply(x)
    x = tf.layers
        .dense({
            name: config.name + '/c_proj',
            units: config.nEmbd,
            inputDim: 4 * config.nEmbd,
            inputShape: [config.blockSize, 4 * config.nEmbd]
        })
        .apply(x)
    x = tf.layers
        .dropout({
            name: config.name + '/drop',
            rate: config.residDrop
        })
        .apply(x)
    return tf.model({ inputs: inputs, outputs: x })
}

function Block(conf) {
    const config = Object.assign({ name: 'h' }, conf)
    const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] })
    let x1, x2
    // Attention
    // Setting epsilon to 1e-5 for LayerNorms to be consistent with PyTorch
    x1 = tf.layers
        .layerNormalization({ name: config.name + '/ln_1', epsilon: 1e-5 })
        .apply(inputs)
    if (config.debug) {
        x1 = LogLayer({ name: config.name + '/ln_1_log' }).apply(x1)
    }
    x1 = new CausalSelfAttention({
        name: config.name + '/attn',
        ...config
    }).apply(x1)
    x1 = tf.layers.add().apply([inputs, x1])
    // MLP
    x2 = tf.layers
        .layerNormalization({ name: config.name + '/ln_2', epsilon: 1e-5 })
        .apply(x1)
    x2 = MLP(Object.assign({}, config, { name: config.name + '/mlp' })).apply(
        x2
    )
    x2 = tf.layers.add().apply([x1, x2])
    return tf.model({ name: config.name, inputs: inputs, outputs: x2 })
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
tf.serialization.registerClass(Range)
