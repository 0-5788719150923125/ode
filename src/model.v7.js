import ODE from './model.v6.js'

/**
 * A better sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 5
        this.units = config.units || 64
        this.experts = config.experts || 7
        this.topK = config.topK || 2
        this.moeDim = config.moeDim || 128
        this.headDim = config.headDim || 256
        this.mlpDim = config.mlpDim || 512
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: 'OriginalDesign/beast'
        })
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

        const encoding = this.ode.layers.RotaryPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .TransientMixtureOfExperts({
                    topK: this.topK,
                    numExperts: this.experts,
                    hiddenDim: this.moeDim,
                    activation: 'swish',
                    expertType: 'SelfAttention',
                    expertArgs: { projection: this.headDim }
                })
                .apply(outputs)

            outputs = this.ode.layers
                .TransientMixtureOfExperts({
                    topK: this.topK,
                    numExperts: this.experts,
                    hiddenDim: this.moeDim,
                    activation: 'swish',
                    expertType: 'GatedLinearMLP',
                    expertArgs: { innerDim: this.mlpDim, activation: 'swish' }
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
