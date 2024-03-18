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
        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        // const positionalEncoder = new PositionalEncodingLayer({
        //     embeddingDim: this.units,
        //     maxSeqLength: this.config.contextLength
        // })

        let state = new PositionalEncodingLayer({
            embeddingDim: this.units,
            maxSeqLength: this.config.contextLength
        }).apply(embeddings)

        // const maskingLayer = this.tf.layers.masking({ maskValue: 0 })
        // const maskedOutput = maskingLayer.apply(state)
        // console.log(maskedOutput)

        // state = positionalEncoder.apply(state)

        for (let i = 0; i < this.layers; i++) {
            const decoder = new TransformerBlock({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads,
                activation: 'swish'
                // mask: maskedOutput
            })
            state = decoder.apply(state)
        }

        const head = this.tf.layers.dense({
            units: this.tokenizer.getLength(),
            activation: 'linear'
        })

        const outputs = head.apply(state)

        this.model = this.tf.model({ inputs, outputs })
    }
}
