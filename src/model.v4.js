import ModelBase from './model.v0.js'
import { CausalSelfAttention, MultiLayerPerceptron, Range } from './layers.js'
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

    setupTokenizer() {
        this.tokenizer = new PretrainedTokenizer()
    }

    build() {
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

        const positionalEmbeddings = this.tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        let outputs = this.tf.layers
            .add()
            .apply([tokenEmbeddings, positionalEmbeddings])

        outputs = this.tf.layers
            .dropout({
                name: 'dropout',
                rate: this.dropout
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = new CausalSelfAttention({
                blockSize: this.config.contextLength,
                units: this.units,
                nHead: this.numHeads,
                dropout: this.dropout,
                bias: false
            }).apply(outputs)

            outputs = new MultiLayerPerceptron({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads,
                dropout: this.dropout,
                activation: 'gelu'
            }).apply(outputs)
        }

        outputs = this.tf.layers
            .layerNormalization({
                name: 'head/ln',
                epsilon: 1e-5
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

        // either sample from the distribution or take the most likely element
        if (temperature !== 1) {
            // scale by desired temperature
            logitsScaled = logitsScaled.div(this.tf.scalar(temperature))
            idxNext = this.tf.multinomial(logitsScaled, 1)
        } else {
            idxNext = logitsScaled.softmax(-1).argMax(-1).expandDims(1)
        }

        this.tf.keep(idxNext)
    })
    return idxNext
}
