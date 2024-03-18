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
        this.size = 256
    }

    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })
        let embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.size,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        let x = new PositionalEncodingLayer({
            embeddingDim: this.size,
            maxSeqLength: this.config.contextLength
        }).apply(embeddings)

        for (let i = 0; i < this.layers; i++) {
            x = new CausalAttentionLayer({ units: this.size }).apply(x)
            x = new TransformerBlock({
                units: this.size
            }).apply(x)
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
