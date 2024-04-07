import ODE from './model.v3.js'

/**
 * An experimental language model.
 * @extends ODE
 */
export default class OscilloscopingDecayedExponent extends ODE {
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

        // const tokenEmbeddings = this.ode.layers
        //     .embedding({
        //         inputDim: this.tokenizer.getLength(),
        //         outputDim: this.units,
        //         embeddingsInitializer: 'glorotUniform'
        //     })
        //     .apply(inputs)

        // const range = this.ode.layers.Range().apply(inputs)

        // const positionalEmbeddings = this.ode.layers
        //     .embedding({
        //         inputDim: this.config.contextLength,
        //         outputDim: this.units,
        //         embeddingsInitializer: 'glorotNormal'
        //     })
        //     .apply(range)

        // let outputs = this.tf.layers
        //     .add()
        //     .apply([tokenEmbeddings, positionalEmbeddings])

        const tokenEmbeddings = this.ode.layers
            .DeterministicEmbedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units
            })
            .apply(inputs)

        let outputs = this.ode.layers
            .SinusoidalPositionalEncoding({
                units: this.units
            })
            .apply(tokenEmbeddings)

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
