import ModelBase from './model.v0.js'
import {
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock
} from './layers.js'

/**
 * A small transformer with multi-head attention and sinusoidal position embeddings.
 * @extends ModelBase
 */
export default class OmniscientDeterministicEngine extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 512
        this.innerDim = this.units * 4
    }

    build() {
        super.build()

        let state

        const inputs = this.tf.input({ shape: [null] })
        const embeddings = this.tf.layers.embedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform',
            maskZero: true
        })

        state = embeddings.apply(inputs)

        const encoder = new SinusoidalPositionalEncoding({
            embeddingDim: this.units,
            maxSeqLength: this.config.contextLength
        })

        state = encoder.apply(state)

        for (let i = 0; i < this.layers; i++) {
            const attention = new MultiHeadAttention({
                numHeads: this.numHeads,
                units: this.units
            })
            state = attention.apply(state)
            const decoder = new TransformerBlock({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads,
                activation: 'swish'
            })
            state = decoder.apply(state)
        }

        const head = this.tf.layers.dense({
            units: this.tokenizer.getLength(),
            activation: 'linear'
        })

        state = head.apply(state)

        this.model = this.tf.model({ inputs, outputs: state })
    }

    async compile() {
        this.lossFunctions = [this.tf.losses.softmaxCrossEntropy]
        this.model.compile({
            optimizer: this.tf.train.adam(
                this.config.learningRate || 1e-3,
                this.config.beta1 || 0.9,
                this.config.beta2 || 0.999,
                this.config.epsilon || 1e-8
            ),
            loss: this.lossFunctions
        })
    }
}
