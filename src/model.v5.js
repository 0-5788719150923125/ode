import ODE from './model.v2.js'

/**
 * A sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.experts = config.experts || 7
        this.topK = config.topK || 2
        this.moeDim = config.moeDim || 128
        this.headDim = config.headDim || 512
        this.mlpDim = config.mlpDim || 1024
    }

    defineTokenizer() {
        super.defineTokenizer({
            model: 'OriginalDesign/twos'
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
            const experts = this.createAttentionExperts()
            const expertOutputs = experts.map((expert) => expert.apply(outputs))

            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    topK: this.topK,
                    numExperts: experts.length,
                    hiddenDim: this.moeDim,
                    activation: 'swish'
                })
                .apply([outputs, ...expertOutputs])

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    innerDim: this.mlpDim,
                    activation: 'swish'
                })
                .apply(outputs)
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
}
