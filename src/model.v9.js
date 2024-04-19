import ODE from './model.v4.js'

/**
 * For CPU-only peers.
 * @extends ODE
 */
export default class ObjectivelyDumbExample extends ODE {
    constructor(config) {
        super(config)
        this.units = 256
    }

    async defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
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

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
