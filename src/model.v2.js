import ModelBase from './model.v0.js'
import { CausalAttentionLayer, ResidualConnectionLayer } from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })
        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: 64,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        const layers = [128, 128, 128, 128]
        let recurrentOutput = embeddings
        layers.forEach((size, i) => {
            const notFirstLayer = i !== 0
            const notLastLayer = i < layers.length - 1
            const layer = this.tf.layers.gru({
                units: size,
                activation: 'softsign',
                kernelInitializer: 'glorotUniform',
                recurrentActivation: 'sigmoid',
                recurrentInitializer: 'orthogonal',
                returnSequences: notLastLayer
            })
            const currentOutput = layer.apply(recurrentOutput)

            if (notFirstLayer && notLastLayer) {
                const residual = new ResidualConnectionLayer()
                recurrentOutput = residual.apply([
                    currentOutput,
                    recurrentOutput
                ])
            } else {
                recurrentOutput = currentOutput
            }

            const norm = this.tf.layers.layerNormalization({
                epsilon: 1e-3
            })
            recurrentOutput = norm.apply(recurrentOutput)

            if (notLastLayer) {
                const attention = new CausalAttentionLayer({
                    units: size,
                    kernelInitializer: 'glorotUniform'
                })
                recurrentOutput = attention.apply(recurrentOutput)
            }
        })

        // Add the final dense layer
        const outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(recurrentOutput)

        this.model = this.tf.model({ inputs, outputs })
    }
}
