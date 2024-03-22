import ModelBase from './model.v0.js'
import {
    DebugLayer,
    GPT2Block,
    Range,
    SinusoidalPositionalEncoding
} from './layers.js'
import { getAdamW } from './optimizers.js'
import PretrainedTokenizer from './tokenizers.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends ModelBase
 */
export default class OriginalDecoderEngine extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 256
        this.dropout = 0.1
    }

    setupTokenizer(vocabSize = 16666, numIterations = 500_000_000) {
        // super.setupTokenizer(vocabSize, numIterations)
        this.tokenizer = new PretrainedTokenizer()
    }

    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })

        const tokenEmbeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = new Range().apply(inputs)

        const positionalEmbeddings = new SinusoidalPositionalEncoding({
            units: this.units,
            sequenceLength: this.config.contextLength
        }).apply(range)

        // const positionalEmbeddings = this.tf.layers
        //     .embedding({
        //         name: 'wpe',
        //         inputDim: this.config.contextLength,
        //         outputDim: this.units,
        //         embeddingsInitializer: 'glorotUniform'
        //     })
        //     .apply(debug)

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
                name: 'gpt' + '/h/' + i,
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
                name: 'gpt' + '/ln_f',
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
                this.config.decayRate || 1e-1
            ),
            loss: this.lossFunctions
        })
    }

    async generate(prompt, temperature = 0.7, length = 20) {
        return await generateText.call(this, prompt, temperature, length)
    }
}

async function generateText(prompt, temperature, maxNewTokens) {
    let inputs = await prepareInputs.call(this, this.tokenizer.encode(prompt))
    for (let step = 0; step < maxNewTokens; step++) {
        const idxNext = generateOnce.call(this, inputs, temperature)
        const idxNew = inputs.concat(idxNext, 1)
        this.tf.dispose(inputs)
        inputs = idxNew
        this.tf.dispose(idxNext)
    }
    const idxArr = await inputs.array()
    this.tf.dispose(inputs)
    return this.tokenizer.decode(idxArr[0])
}

function prepareInputs(inputs) {
    this.tf.tidy(() => {
        // Check if idx is a tensor or an array
        if (inputs instanceof this.tf.Tensor) {
            inputs = inputs.clone()
        } else {
            inputs = this.tf.tensor(inputs)
        }
        // Check data type
        if (inputs.dtype !== 'int32') {
            inputs = inputs.toInt()
        }
        // If the shape of idx is 1D, we need to add a dimension
        if (inputs.shape.length === 1) {
            inputs = inputs.expandDims(0)
        }
        this.tf.keep(inputs)
        // keep idx from deletion
    })
    return inputs
}

function generateOnce(idx, temperature) {
    let idxNext
    this.tf.tidy(() => {
        const block_size = this.model.inputs[0].shape[1]
        const idxCond =
            idx.shape[1] <= block_size
                ? idx
                : idx.slice([0, -block_size], [-1, -1])
        // Forward the model to get the logits for the index in the sequence
        const logits = this.model.predict(idxCond)
        // pluck the logits at the final step
        let logitsScaled = logits
            .slice([0, idx.shape[1] - 1, 0])
            .reshape([logits.shape[0], logits.shape[2]])

        if (temperature !== 1) {
            // scale by desired temperature
            logitsScaled = logitsScaled.div(this.tf.scalar(temperature))
        }

        // either sample from the distribution or take the most likely element
        if (temperature > 0) {
            idxNext = this.tf.multinomial(logitsScaled, 1)
        } else {
            idxNext = logitsScaled.softmax(-1).argMax(-1).expandDims(1)
        }
        this.tf.keep(idxNext)
    })
    return idxNext
}
