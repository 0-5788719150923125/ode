import ODE from './model.v4.js'

/**
 * A state space model.
 * @extends ODE
 */
export default class OrthogonalDepthwiseEntanglement extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.innerDim = config.innerDim || 1024
        this.decayFactor = config.decayFactor || 1.0
        this.activation = config.activation || 'softsign'
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer(config)
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform'
        })

        let outputs = embeddings.apply(inputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .StateSpace({
                    units: this.units,
                    innerDim: this.innerDim,
                    returnSequences: true,
                    decayFactor: this.decayFactor,
                    activation: this.activation
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
