import ODE from './model.v3.js'

/**
 * A better sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.headDim = config.headDim || 1024
        this.mlpDim = config.mlpDim || 768
        this.numExperts = config.numExperts || 3
        this.topK = config.topK || 2
        this.switchingDim = config.switchingDim || 512
        this.temperature = config.temperature || 0.8
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-consistent-v1'
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
                .AdaptiveMixtureOfExperts({
                    topK: this.topK,
                    switchingDim: this.switchingDim,
                    activation: 'swish',
                    temperature: this.temperature,
                    experts: this.createMLPExperts(outputs.shape)
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    createMLPExperts(inputShape) {
        return Array(this.numExperts)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'MultiLayerPerceptron',
                    inputShape,
                    hiddenDim: this.mlpDim,
                    activation: 'swish'
                })
            })
    }
}
