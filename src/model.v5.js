import OpportunisticDialogueEncoder from './model.v4.js'

/**
 * A state space model.
 * @extends OpportunisticDialogueEncoder
 */
export default class ObservableDataEncryption extends OpportunisticDialogueEncoder {
    constructor(config) {
        super(config)
        this.layers = 4
        this.units = 256
        this.innerDim = this.units * 4
        this.chunkSize = 8
        this.epsilon = 1e-5
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .StructuredStateSpace({
                    units: this.units,
                    innerDim: this.innerDim,
                    chunkSize: this.chunkSize,
                    epsilon: this.epsilon,
                    returnSequences: true
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
