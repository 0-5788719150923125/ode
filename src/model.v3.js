import ODE from './model.v2.js'

/**
 * The simplest transformer.
 * @extends ODE
 */
export default class OptimalDecisionEngine extends ODE {
    constructor(config) {
        const defaults = {
            layers: 3,
            units: 256,
            learningRate: 1e-4,
            weightDecay: 1e-5
        }
        super({ ...defaults, ...config })
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
            outputDim: this.config.units,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .SelfAttention({
                    hiddenDim: this.config.units * 4
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    hiddenDim: this.config.units * 4,
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
