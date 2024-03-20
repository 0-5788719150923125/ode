import ModelBase from './model.v0.js'
import { SinusoidalPositionalEncoding } from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 3
        this.numHeads = 8
        this.units = 128
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
            const layer = this.tf.layers.gru({
                units: this.units,
                activation: 'softsign',
                kernelInitializer: 'glorotUniform',
                recurrentActivation: 'sigmoid',
                recurrentInitializer: 'orthogonal',
                returnSequences: true
            })
            state = layer.apply(state)
        }

        const head = this.tf.layers.timeDistributed({
            layer: this.tf.layers.dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
        })

        const outputs = head.apply(state)

        this.model = this.tf.model({ inputs, outputs })
    }
}
