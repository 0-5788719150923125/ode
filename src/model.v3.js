import ODE from './model.v2.js'

/**
 * The simplest transformer.
 * @extends ODE
 */
export default class OptimalDecisionEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.learningRate = 0.0001
        this.weightDecay = 0.001
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
                    hiddenDim: this.units * 4
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    hiddenDim: this.units * 4,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        return [this.ode.schedulers.constantScheduler(this.learningRate)]
    }
}
