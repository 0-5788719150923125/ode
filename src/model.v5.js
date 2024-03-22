import ModelBase from './model.v0.js'
import {
    CausalAttentionLayer,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock
} from './layers.js'

/**
 * A small transformer with multi-head attention and sinusoidal position embeddings.
 * @extends ModelBase
 */
export default class OmniscientDeterministicEnsemble extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 512
        this.innerDim = this.units * 4
    }

    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })
        const embeddings = this.tf.layers.embedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform',
            maskZero: true
        })

        let outputs = embeddings.apply(inputs)

        const encoder = new SinusoidalPositionalEncoding({
            embeddingDim: this.units,
            maxSeqLength: this.config.contextLength
        })

        outputs = encoder.apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            const attention = new CausalAttentionLayer({
                // numHeads: this.numHeads,
                units: this.units
            })
            outputs = attention.apply(outputs)
            const decoder = new TransformerBlock({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads,
                activation: 'swish'
            })
            outputs = decoder.apply(outputs)
        }

        const head = this.tf.layers.dense({
            units: this.tokenizer.getLength(),
            activation: 'linear'
        })

        outputs = head.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    async compile() {
        this.lossFunctions = [this.tf.losses.softmaxCrossEntropy]
        this.model.compile({
            optimizer: this.tf.train.adam(
                this.config.learningRate || 1e-3,
                this.config.beta1 || 0.9,
                this.config.beta2 || 0.999,
                this.config.epsilon || 1e-7
            ),
            loss: this.lossFunctions
        })
    }
}
