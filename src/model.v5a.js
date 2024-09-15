import ODE from './model.v3.js'

/**
 * A kinda-sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super({
            layers: 3,
            units: 256,
            headDim: 1024,
            mlpDim: 512,
            numExperts: 3,
            topK: 2,
            switchingDim: 512,
            temperature: 0.8,
            ...config
        })
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-clean-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.config.units,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.config.layers; i++) {
            let normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: false, useBias: false })
                .apply(outputs)
            const attnOutputs = this.ode.layers
                .SelfAttention({
                    hiddenDim: this.config.headDim
                })
                .apply(normalized)

            outputs = this.ode.layers
                .ResidualConnection()
                .apply([attnOutputs, outputs])

            normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: false, useBias: false })
                .apply(outputs)

            const experts = this.createExperts()
            const expertOutputs = experts.map((expert) =>
                expert.apply(normalized)
            )

            const ffdOutputs = this.ode.layers
                .SparseMixtureOfExperts({
                    topK: this.config.topK,
                    numExperts: experts.length,
                    switchingDim: this.config.switchingDim,
                    activation: 'swish',
                    temperature: this.config.temperature
                })
                .apply([normalized, ...expertOutputs])

            outputs = this.ode.layers
                .ResidualConnection()
                .apply([ffdOutputs, outputs])
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    createExperts() {
        const experts = []
        for (let i = 0; i < this.config.numExperts; i++) {
            experts.push(
                this.ode.layers.MultiLayerPerceptron({
                    hiddenDim: this.config.mlpDim,
                    activation: 'mish'
                })
            )
        }
        return experts
    }
}
