import ODE from './model.v8.js'

/**
 * A model that's still in development.
 * @extends ODE
 */
export default class OscillatingDiagonalEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 512
        this.projection = config.projection || 64
        this.queries = config.queries || 8
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
                    projection: this.projection,
                    queries: this.queries
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearUnit({
                    innerDim: this.units * 4,
                    activation: 'selu'
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
