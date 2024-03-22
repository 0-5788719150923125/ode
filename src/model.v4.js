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
                name: 'wte',
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
                name: 'wpe',
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
                name: 'head',
                units: this.tokenizer.getLength(),
                inputDim: this.units,
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
                this.config.decayRate || 1e-4
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
