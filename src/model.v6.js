import ODE from './model.v3.js'

/**
 * An experimental, deterministic language model with next to 0 trainable parameters.
 * @extends ODE
 */
export default class OscilloscopingDecayedExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = 9
        this.units = 256
        this.maxDecisions = 9
        this.kernelSize = 6
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

        const embeddings = this.ode.layers.SharedEmbedding({
            vocabSize: this.tokenizer.getLength(),
            embeddingDim: this.units
        })

        const encoding = this.ode.layers.RotaryPositionalEncoding({
            blockSize: this.config.contextLength,
            units: this.units
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

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
