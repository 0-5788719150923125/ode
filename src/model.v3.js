import ModelBase from './model.v0.js'
import { CausalAttentionLayer, TransformerEncoderBlock } from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    build() {
        super.build()

        // Add the embedding layer as the first layer
        const inputs = this.tf.input({ shape: [null] })
        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: 64,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        let x = embeddings
        for (let i = 0; i < 3; i++) {
            // For simplicity, using 3 Transformer blocks
            x = new TransformerEncoderBlock({
                units: 64
            }).apply(x)
        }

        // Add the final dense layer
        const finalDense = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(x)

        this.model = this.tf.model({ inputs: inputs, outputs: finalDense })
    }
}
