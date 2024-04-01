import OpportunisticDialogueEncoder from './model.v4.js'
import { randomString } from './utils.js'

/**
 * A state space model.
 * @extends OpportunisticDialogueEncoder
 */
export default class ObservableDataEncryption extends OpportunisticDialogueEncoder {
    constructor(config) {
        super(config)
        this.layers = 3
        this.units = 64
        this.innerDim = this.units * 4
    }

    async defineTokenizer() {
        await super.defineTokenizer({
            model: 'OriginalDesign/frame'
        })
    }

    defineBuild() {
        const inputs = this.tf.input({
            name: `inn-${randomString()}`,
            shape: [null]
        })

        let outputs = this.tf.layers
            .embedding({
                name: `emb-${randomString()}`,
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .RotaryPositionalEncoding({
                blockSize: this.config.contextLength,
                units: this.units
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            const isLastLayer = i < this.layers - 1
            outputs = this.ode.layers
                .StateSpace({
                    units: this.units,
                    innerDim: this.innerDim,
                    returnSequences: isLastLayer
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .dense({
                name: `out-${randomString()}`,
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
