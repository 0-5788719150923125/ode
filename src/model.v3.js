import ModelBase from './model.v0.js'
import {
    CausalAttentionLayer,
    PositionalEncodingLayer,
    TransformerBlock
} from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 3
        this.units = 256
    }

    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })
        let embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        let x = new PositionalEncodingLayer({
            embeddingDim: this.units,
            maxSeqLength: this.config.contextLength
        }).apply(embeddings)

        for (let i = 0; i < this.layers; i++) {
            x = new CausalAttentionLayer({ units: this.units }).apply(x)
            x = this.tf.layers
                .dense({
                    units: this.units * 3,
                    activation: 'swish',
                    kernelInitializer: 'glorotUniform'
                })
                .apply(x)
            x = this.tf.layers
                .dense({
                    units: this.units,
                    activation: 'linear',
                    kernelInitializer: 'glorotUniform'
                })
                .apply(x)
            // x = new TransformerBlock().apply(x)
            x = this.tf.layers.layerNormalization().apply(x)
        }

        const pooled = this.tf.layers.globalAveragePooling1d().apply(x)

        const outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(pooled)

        this.model = this.tf.model({ inputs, outputs })
    }
}
