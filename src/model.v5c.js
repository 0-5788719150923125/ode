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
            numExperts: 3,
            topK: 2,
            switchingDim: 256,
            headDim: 1024,
            mlpDim: 512,
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
                .RMSNorm({ elementwiseAffine: true, useBias: false })
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
                .RMSNorm({ elementwiseAffine: true, useBias: false })
                .apply(outputs)
            const ffdOutputs = this.ode.layers
                .SparseMixtureOfExpertsMLP({
                    topK: this.config.topK,
                    numExperts: this.config.numExperts,
                    switchingDim: this.config.switchingDim,
                    activation: 'mish',
                    temperature: this.config.temperature,
                    mlpDim: this.config.mlpDim
                })
                .apply(normalized)

            outputs = this.ode.layers
                .ResidualConnection()
                .apply([ffdOutputs, outputs])
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }
}
