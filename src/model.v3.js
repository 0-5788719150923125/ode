import ModelBase from './model.v0.js'
import {
    LastTokenSelectionLayer,
    PositionalEncodingLayer,
    TransformerBlock
} from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    constructor(config) {
        super(config)
        this.layers = 3
        this.numHeads = 4
        this.units = 256
        this.innerDim = this.units * 3
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

        let state = embeddings.apply(inputs)

        const positionalEncoder = new PositionalEncodingLayer({
            embeddingDim: this.units,
            maxSeqLength: this.config.contextLength
        })

        state = positionalEncoder.apply(state)

        for (let i = 0; i < this.layers; i++) {
            const decoder = new TransformerBlock({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads
            })
            state = decoder.apply(state)
        }

        const selector = new LastTokenSelectionLayer()
        state = selector.apply(state)

        const head = this.tf.layers.dense({
            units: this.tokenizer.getLength(),
            activation: 'linear'
        })

        const outputs = head.apply(state)

        this.model = this.tf.model({ inputs, outputs })
    }
}
