import ModelBase from './model.v0.js'
import {
    CausalAttentionLayer,
    PositionalEncodingLayer,
    TransformerBlock
} from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    build() {
        super.build()

        const layers = 3
        const size = 256

        // Add the embedding layer as the first layer
        const inputs = this.tf.input({ shape: [null] })
        let embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: size,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        let x = embeddings
        // x = new PositionalEncodingLayer({
        //     embeddingDim: size,
        //     maxSeqLength: this.config.contextLength
        // }).apply(x)

        for (let i = 0; i < layers; i++) {
            x = new CausalAttentionLayer({ units: size }).apply(x)
            x = new TransformerBlock({
                units: size
            }).apply(x)
            x = this.tf.layers.layerNormalization().apply(x)
        }

        const pooled = this.tf.layers.globalAveragePooling1d().apply(x)

        // Add the final dense layer
        const outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(pooled)

        this.model = this.tf.model({ inputs, outputs })
    }
}
