import ODE from './model.v8.js'

/**
 * In development.
 * @extends ODE
 */
export default class OscillatingDistillationExperiment extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 2
        this.units = config.units || 333
        this.hiddenDim = config.hiddenDim || this.units * 6
        this.projection = config.projection || 111
        this.queries = config.queries || 33
        this.dropout = config.dropout || 0.333
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform',
            dropout: this.dropout
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .MultiQueryAttention({
                    projection: this.projection,
                    queries: this.queries,
                    dropout: this.dropout
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearUnit({
                    innerDim: this.hiddenDim,
                    activation: 'selu',
                    dropout: this.dropout
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
