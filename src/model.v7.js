import ODE from './model.v3.js'

/**
 * A mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = 3
        this.units = 256
        this.experts = 5
        this.topK = 2
        this.moeDim = 128
    }

    defineTokenizer() {
        super.defineTokenizer({
            model: 'OriginalDesign/word'
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

            // outputs = this.ode.layers
            //     .MixtureOfExperts({
            //         numExperts: experts.length,
            //         hiddenDim: this.moeDim,
            //         activation: 'swish'
            //     })
            //     .apply([outputs, ...expertOutputs])

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
                    innerDim: this.units * 4,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.learningRate = 0.001
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    createAttentionExperts() {
        const experts = []
        for (let i = 0; i < this.experts; i++) {
            experts.push(
                this.ode.layers.SelfAttention({
                    projection: this.units * 4
                })
            )
        }
        return experts
    }
}
