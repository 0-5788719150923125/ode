import ModelBase from './model.v0.js'
import { GPT2Block, Range } from './layers.js'
import { getAdamW } from './optimizers.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends ModelBase
 */
export default class OriginalDecoderEngine extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 512
        this.dropout = 0.1
    }

    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })

        const tokenEmbeddings = this.tf.layers
            .embedding({
                // name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform',
                embeddingsRegularizer: null,
                activityRegularizer: null
            })
            .apply(inputs)

        const range = new Range().apply(inputs)

        const positionalEmbeddings = this.tf.layers
            .embedding({
                // name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        let x = this.tf.layers
            .add()
            .apply([tokenEmbeddings, positionalEmbeddings])

        x = this.tf.layers
            .dropout({
                name: 'drop',
                rate: this.dropout
            })
            .apply(x)

        for (let i = 0; i < this.layers; i++) {
            x = GPT2Block({
                name: 'block' + '/h/' + i,
                nLayer: this.layers,
                nHead: this.numHeads,
                nEmbd: this.units,
                blockSize: this.config.contextLength,
                dropout: this.dropout,
                bias: false
            }).apply(x)
        }
        x = this.tf.layers
            .layerNormalization({
                name: 'block' + '/ln_f',
                epsilon: 1e-5
            })
            .apply(x)

        x = this.tf.layers
            .dense({
                // name: 'lm_head',
                units: this.tokenizer.getLength(),
                inputDim: this.units,
                // inputShape: [this.config.contextLength, this.units],
                useBias: false
            })
            .apply(x)

        this.model = this.tf.model({ inputs: inputs, outputs: x })
    }

    async compile() {
        this.lossFunctions = [this.tf.losses.softmaxCrossEntropy]
        this.model.compile({
            optimizer: getAdamW(
                this.model,
                this.config.learningRate || 1e-3,
                this.config.beta1 || 0.9,
                this.config.beta2 || 0.999,
                this.config.epsilon || 1e-7,
                this.config.decayRate || 1e-2
            ),
            loss: this.lossFunctions
        })
    }

    async generate(seed, temperature = 0.7, length = 20) {
        const inputs = this.tokenizer.encode(seed)
        const idx = await generate(this.model, [inputs], {
            maxNewTokens: length,
            temperature: temperature
        })
        return this.tokenizer.decode(idx[0])
    }
}

import tf from '@tensorflow/tfjs'

// class CausalSelfAttentionBase extends tf.layers.Layer {
//     constructor(config, i) {
//         super(config)
//         this.config = config
//         this.blockSize = config.blockSize
//         this.nEmbd = config.nEmbd
//         this.nHead = config.nHead
//         this.dropout = config.dropout
//         this.i = i
//         this.mask = tf.linalg.bandPart(
//             tf.ones([config.blockSize, config.blockSize]),
//             -1,
//             0
//         )
//     }

//     computeOutputShape(inputShape) {
//         // Input here is already passed through a dense layer
//         // It's shape is [B, T, 3 * nEmbd]
//         // 3 there is for k, q, v (same as in MinGPT)
//         // The output is [B, T, nEmbd]
//         return [null, this.blockSize, this.nEmbd]
//     }

//     getConfig() {
//         const config = super.getConfig()
//         return Object.assign({}, config, this.config)
//     }

//     call(input, kwargs) {
//         return tf.tidy(() => {
//             // Take into account that the input can be an array of tensors
//             if (Array.isArray(input)) {
//                 input = input[0]
//             }
//             this.invokeCallHook(input, kwargs)

//             // split() in TFJS requires a constant value for n splits
//             // split() in Pytorch requires the size of each split
//             let [q, k, v] = tf.split(input, 3, -1)
//             const [B, T, C] = k.shape
//             const splitHeads = (x) =>
//                 tf.transpose(
//                     tf.reshape(x, [B, T, this.nHead, C / this.nHead]),
//                     [0, 2, 1, 3]
//                 )
//             q = splitHeads(q)
//             k = splitHeads(k)
//             v = splitHeads(v)

//             // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
//             let att = tf.mul(
//                 tf.matMul(q, k, false, true),
//                 tf.div(
//                     1,
//                     tf.sqrt(tf.cast(k.shape[k.shape.length - 1], 'float32'))
//                 )
//             )
//             att = tf.add(att, tf.mul(tf.sub(1, this.mask), -1e9))
//             att = tf.softmax(att, -1)
//             att = kwargs['training'] ? tf.dropout(att, this.dropout) : att

//             let y = tf.matMul(att, v)
//             y = tf.transpose(y, [0, 2, 1, 3])
//             y = tf.reshape(y, [B, T, C])

//             return y
//         })
//     }

//     static get className() {
//         return 'CausalSelfAttentionBase'
//     }
// }
// tf.serialization.registerClass(CausalSelfAttentionBase)

// function CausalSelfAttentionMixed(conf) {
//     const config = Object.assign({ name: 'attn' }, conf)
//     const csa = new CausalSelfAttentionBase(config)
//     const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] })
//     let att
//     att = tf.layers
//         .dense({
//             name: config.name + '/c_attn',
//             units: 3 * config.nEmbd,
//             inputDim: config.nEmbd,
//             inputShape: [config.blockSize, config.nEmbd],
//             useBias: config.bias
//         })
//         .apply(inputs)
//     att = csa.apply(att)
//     att = tf.layers
//         .dense({
//             name: config.name + '/proj',
//             units: config.nEmbd,
//             inputDim: config.nEmbd,
//             inputShape: [config.blockSize, config.nEmbd],
//             useBias: config.bias
//         })
//         .apply(att)
//     att = tf.layers
//         .dropout({
//             name: config.name + '/drop',
//             rate: config.dropout
//         })
//         .apply(att)
//     return tf.model({ inputs: inputs, outputs: att })
// }

// function GPT(conf) {
//     const configDefaults = {
//         name: 'transformer',
//         bias: true,
//         debug: false,
//         tokEmb: true,
//         lmHead: true
//     }
//     const configModels = {
//         gpt2: {
//             nLayer: 12,
//             nHead: 12,
//             nEmbd: 768,
//             vocabSize: 50257,
//             blockSize: 1024
//         },
//         'gpt2-medium': {
//             nLayer: 24,
//             nHead: 16,
//             nEmbd: 1024,
//             vocabSize: 50257,
//             blockSize: 1024
//         },
//         'gpt2-large': {
//             nLayer: 36,
//             nHead: 20,
//             nEmbd: 1280,
//             vocabSize: 50257,
//             blockSize: 1024
//         },
//         'gpt2-xl': {
//             nLayer: 48,
//             nHead: 25,
//             nEmbd: 1600,
//             vocabSize: 50257,
//             blockSize: 1024
//         },
//         'gpt-mini': { nLayer: 6, nHead: 6, nEmbd: 192 },
//         'gpt-micro': { nLayer: 4, nHead: 4, nEmbd: 128 },
//         'gpt-nano': { nLayer: 3, nHead: 3, nEmbd: 48 }
//     }
//     // Check if modelType is present in conf
//     if (conf.modelType) {
//         // If so, check if it's valid
//         if (!Object.keys(configModels).includes(conf.modelType)) {
//             throw new Error(`Invalid modelType: ${conf.modelType}`)
//         }
//         // If valid, merge modelConfig with configDefaults
//         const modelConfig = configModels[conf.modelType]
//         Object.assign(configDefaults, modelConfig)
//     }

//     const config = Object.assign({}, configDefaults, conf)

//     const inputs = tf.input({ shape: [null] })

//     const tokEmb = config.tokEmb
//         ? tf.layers
//               .embedding({
//                   name: config.name + '/wte',
//                   inputDim: config.vocabSize,
//                   outputDim: config.nEmbd,
//                   embeddingsInitializer: 'zeros',
//                   embeddingsRegularizer: null,
//                   activityRegularizer: null
//               })
//               .apply(inputs)
//         : inputs

//     const range = Range().apply(inputs)
//     let posEmb = tf.layers
//         .embedding({
//             name: config.name + '/wpe',
//             inputDim: config.blockSize,
//             outputDim: config.nEmbd,
//             embeddingsInitializer: 'zeros'
//         })
//         .apply(range)

//     let x
//     x = tf.layers.add().apply([tokEmb, posEmb])
//     x = tf.layers
//         .dropout({
//             name: 'drop',
//             rate: config.embdDrop
//         })
//         .apply(x)

//     for (let i = 0; i < config.nLayer; i++) {
//         x = Block(
//             Object.assign({}, config, { name: config.name + '/h/' + i })
//         ).apply(x)
//     }
//     x = tf.layers
//         .layerNormalization({ name: config.name + '/ln_f', epsilon: 1e-5 })
//         .apply(x)

//     if (config.lmHead) {
//         x = tf.layers
//             .dense({
//                 name: 'lm_head',
//                 units: config.vocabSize,
//                 inputDim: config.nEmbd,
//                 inputShape: [config.blockSize, config.nEmbd],
//                 useBias: false
//             })
//             .apply(x)
//     }
//     return tf.model({ inputs: inputs, outputs: x })
// }

// const defaultGenerateConfig = {
//     maxNewTokens: 20,
//     temperature: 1.0,
//     doSample: false,
//     topK: null
// }

function prepareIdx(idx) {
    tf.tidy(() => {
        // Check if idx is a tensor or an array
        if (idx instanceof tf.Tensor) {
            idx = idx.clone()
        } else {
            idx = tf.tensor(idx)
        }
        // Check data type
        if (idx.dtype !== 'int32') {
            idx = idx.toInt()
        }
        // If the shape of idx is 1D, we need to add a dimension
        if (idx.shape.length === 1) {
            idx = idx.expandDims(0)
        }
        tf.keep(idx)
        // keep idx from deletion
    })
    return idx
}

function generateOnce(model, idx, config) {
    let idxNext
    let timePerToken = performance.now()
    tf.tidy(() => {
        const block_size = model.inputs[0].shape[1]
        const idxCond =
            idx.shape[1] <= block_size
                ? idx
                : idx.slice([0, -block_size], [-1, -1])
        // Forward the model to get the logits for the index in the sequence
        const logits = model.predict(idxCond)
        timePerToken = performance.now() - timePerToken
        // pluck the logits at the final step and scale by desired temperature
        const logitsScaled = logits
            .slice([0, idx.shape[1] - 1, 0])
            .reshape([logits.shape[0], logits.shape[2]])
            .div(tf.scalar(config.temperature))
        // TODO: topK sampling
        // apply softmax to convert logits to (normalized) probabilities
        const probs = logitsScaled.softmax(-1)
        // either sample from the distribution or take the most likely element
        if (config.doSample) {
            idxNext = tf.multinomial(probs, 1)
        } else {
            idxNext = probs.argMax(-1)
            idxNext = idxNext.expandDims(1)
        }
        tf.keep(idxNext)
    })
    return {
        idxNext,
        timePerToken
    }
}

async function generate(model, idx, conf, callback) {
    const defaultGenerateConfig = {
        maxNewTokens: 20,
        temperature: 0,
        doSample: false,
        topK: null
    }
    const config = Object.assign({}, defaultGenerateConfig, conf)
    if (config.temperature > 0) {
        config.doSample = true
    }
    idx = await prepareIdx(idx)
    for (let step = 0; step < config.maxNewTokens; step++) {
        const { idxNext, timePerToken } = generateOnce(model, idx, config)
        const idxNew = idx.concat(idxNext, 1)
        tf.dispose(idx)
        idx = idxNew
        const idxNextArr = await idxNext.array()
        tf.dispose(idxNext)
        if (callback) {
            await callback({ idxNext: idxNextArr, timePerToken: timePerToken })
        }
    }
    const idxArr = await idx.array()
    tf.dispose(idx)
    return idxArr
}

// const GPTModel = (config) => new GPTModel_(config)
// class GPTModel_ {
//     constructor(config) {
//         this.config = config
//         this.model = GPT(config)
//     }

//     async load(modelPath) {
//         await this.model.loadWeights(modelPath)
//     }

//     async save(modelPath) {
//         await this.model.save(modelPath)
//     }

//     apply(inputs) {
//         return this.model.apply(inputs)
//     }

//     predict(inputs) {
//         return this.model.predict(inputs)
//     }
// }

// const GPTLMHeadModel = (config) => new GPTLMHeadModel_(config)
// class GPTLMHeadModel_ extends GPTModel_ {
//     constructor(config) {
//         super(config)
//     }

//     async train(dataset, config) {
//         await train(this.model, dataset, config)
//     }

//     async generate() {
//         return await generate(this.model, ...arguments)
//     }

//     generateSync() {
//         return generateSync(this.model, ...arguments)
//     }
// }

// module.exports = {
//     GELU,
//     CausalSelfAttention,
//     CausalSelfAttentionMixed,
//     MLP,
//     Block,
//     GPT,
//     GPTModel,
//     GPTLMHeadModel,
//     generate,
//     generateSync
// }
