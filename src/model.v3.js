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

        // const layers = [128, 128, 128]
        // let recurrentOutput = embeddings
        let x = embeddings
        for (let i = 0; i < 3; i++) {
            // For simplicity, using 3 Transformer blocks
            x = new TransformerEncoderBlock({
                units: 64
            }).apply(x)
        }
        // layers.forEach((size, i) => {
        //     const notLastLayer = i < layers.length - 1
        //     const layer = this.tf.layers.gru({
        //         units: size,
        //         activation: 'softsign',
        //         kernelInitializer: 'glorotUniform',
        //         recurrentActivation: 'sigmoid',
        //         recurrentInitializer: 'orthogonal',
        //         returnSequences: notLastLayer
        //     })
        //     recurrentOutput = layer.apply(recurrentOutput)

        //     const norm = this.tf.layers.layerNormalization({
        //         epsilon: 1e-3
        //     })
        //     recurrentOutput = norm.apply(recurrentOutput)

        //     // if (notLastLayer) {
        //     //     const attention = new CausalAttentionLayer({
        //     //         units: 128,
        //     //         kernelInitializer: 'glorotUniform'
        //     //     })
        //     //     recurrentOutput = attention.apply(recurrentOutput)
        //     // }
        // })

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
