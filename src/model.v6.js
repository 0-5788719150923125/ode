import ODE from './model.v3.js'

/**
 * An experimental, deterministic language model with next to 0 trainable parameters.
 * @extends ODE
 */
export default class OscillatingDecayedExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = 23
        this.units = 64
        this.routingIterations = 27
        this.decayRate = 0.9
    }

    async defineTokenizer() {
        await super.defineTokenizer({
            model: 'OriginalDesign/frame'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.DeterministicEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding({
            units: this.units
        })

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .Vectorrent({
                    routingIterations: this.routingIterations,
                    decayRate: this.decayRate
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
