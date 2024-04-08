import ODE from './model.v4.js'

/**
 * An experimental, deterministic language model with next to 0 trainable parameters.
 * @extends ODE
 */
export default class OscilloscopingDecayedExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = 9
        this.units = 512
        this.embeddings = 64
        this.maxDecisions = 9
        this.kernelSize = 6
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.embedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.embeddings,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.RotaryPositionalEncoding({
            blockSize: this.config.contextLength
        })

        let outputs = encoding.apply(embeddings.apply(inputs))

        outputs = this.ode.layers
            .Expansion({
                units: this.units
            })
            .apply(outputs)

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
            .bottleneck({
                units: this.embeddings,
                activation: 'softsign'
            })
            .apply(outputs)

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
