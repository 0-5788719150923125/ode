import OriginalDecoderEngine from './model.v4.js'

/**
 * A small transformer with multi-head attention and sinusoidal position embeddings.
 * @extends ModelBase
 */
export default class OmniscientDeterministicEnsemble extends OriginalDecoderEngine {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 256
        this.innerDim = this.units * 4
        this.dropout = 0
    }

    build() {
        const inputs = this.tf.input({ shape: [null] })

        const embeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.ode.layers.Range().apply(inputs)

        const encoding = this.ode.layers
            .SinusoidalPositionalEncoding({
                units: this.units,
                reverse: false
            })
            .apply(range)

        let outputs = this.tf.layers.add().apply([embeddings, encoding])

        for (let i = 0; i < this.layers; i++) {
            // outputs = this.ode.layers.CausalSelfAttention({
            //     blockSize: this.config.contextLength,
            //     units: this.units,
            //     numHeads: this.numHeads,
            //     dropout: this.dropout,
            //     bias: false
            // }).apply(outputs)

            outputs = this.ode.layers
                .MultiHeadAttention({
                    units: this.units,
                    numHeads: this.numHeads,
                    dropout: this.dropout
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    units: this.units,
                    innerDim: this.innerDim,
                    numHeads: this.numHeads,
                    dropout: this.dropout,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .layerNormalization({
                epsilon: 1e-5
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
