import ODE from './model.v8.js'

/**
 * A model that's still in development.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.projection = config.projection || 1024
        this.queries = config.queries || 4
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

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .MultiQueryAttention({
                    units: this.units,
                    projection: this.projection,
                    queries: this.queries
                })
                .apply(outputs)

            outputs = this.ode.layers
                .KolmogorovArnoldNetwork({
                    units: 16,
                    degree: 4
                })
                .apply(outputs)

            // outputs = this.ode.layers
            //     .GatedLinearUnit({
            //         units: this.units,
            //         innerDim: this.units * 4,
            //         activation: 'aptx'
            //     })
            //     .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
