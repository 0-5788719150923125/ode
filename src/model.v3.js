import ModelBase from './model.v0.js'
import { CausalAttentionLayer, TransformerBlock } from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    build() {
        super.build()

        // Add the embedding layer as the first layer
        const inputs = this.tf.input({ shape: [null] })
        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: 512,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        let x = embeddings
        for (let i = 0; i < 3; i++) {
            x = new CausalAttentionLayer({ units: 512 }).apply(x)
            x = new TransformerBlock({
                units: 512
            }).apply(x)
            x = this.tf.layers.layerNormalization().apply(x)
        }

        const pooled = this.tf.layers.globalAveragePooling1d().apply(x)

        // Add the final dense layer
        const finalDense = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(pooled)

        this.model = this.tf.model({ inputs: inputs, outputs: finalDense })
    }
}
