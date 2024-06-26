import ODE from './model.v6.js'

/**
 * A sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 4
        this.units = config.units || 128
        this.experts = config.experts || 7
        this.topK = config.topK || 2
        this.moeDim = config.moeDim || 256
        this.headDim = config.headDim || 256
        this.mlpDim = config.mlpDim || 512
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: 'OriginalDesign/thrice'
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

        const attentionExperts = this.createAttentionExperts()
        const feedforwardExperts = this.createAttentionExperts()

        for (let i = 0; i < this.layers; i++) {
            const attentionOutputs = attentionExperts.map((expert) =>
                expert.apply(outputs)
            )

            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    topK: this.topK,
                    numExperts: attentionExperts.length,
                    hiddenDim: this.moeDim,
                    activation: 'mish'
                })
                .apply([outputs, ...attentionOutputs])

            const feedforwardOutputs = feedforwardExperts.map((expert) =>
                expert.apply(outputs)
            )

            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    topK: this.topK,
                    numExperts: feedforwardExperts.length,
                    hiddenDim: this.moeDim,
                    activation: 'mish'
                })
                .apply([outputs, ...feedforwardOutputs])
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    createAttentionExperts() {
        const experts = []
        for (let i = 0; i < this.experts; i++) {
            experts.push(
                this.ode.layers.SelfAttention({
                    projection: this.headDim
                })
            )
        }
        return experts
    }

    createFeedforwardExperts() {
        const experts = []
        for (let i = 0; i < this.experts; i++) {
            experts.push(
                this.ode.layers.GatedLinearMLP({
                    innerDim: this.mlpDim,
                    activation: 'mish'
                })
            )
        }
        return experts
    }
}
