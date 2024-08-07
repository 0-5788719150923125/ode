import ODE from './model.v3.js'

/**
 * A kinda-sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.numExperts = config.numExperts || 3
        this.topK = config.topK || 2
        this.switchingDim = config.switchingDim || 256
        this.headDim = config.headDim || 1024
        this.mlpDim = config.mlpDim || 512
        this.temperature = 0.8
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-clean-v1'
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

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SelfAttention({
                    hiddenDim: this.headDim
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SparseMixtureOfExpertsMLP({
                    topK: this.topK,
                    numExperts: this.numExperts,
                    switchingDim: this.switchingDim,
                    activation: 'mish',
                    temperature: this.temperature,
                    mlpDim: this.mlpDim
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
