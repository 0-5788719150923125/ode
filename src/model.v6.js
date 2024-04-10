import ODE from './model.v4.js'

/**
 * An experimental, fast language model with limited memory footprint.
 * @extends ODE
 */
export default class OscilloscopingDecayedExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = 12
        this.units = 256
        this.maxDecisions = 9
        this.kernelSize = 9
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.embedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.RotaryPositionalEncoding({
            blockSize: this.config.contextLength
        })

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .Vectorrent({
                    maxDecisions: this.maxDecisions,
                    kernelSize: this.kernelSize,
                    units: this.units
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
