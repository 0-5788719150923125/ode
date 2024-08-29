import ODE from './model.v3.js'

/**
 * A better sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        const defaults = {
            layers: 3,
            units: 256,
            headDim: 1024,
            mlpDim: 768,
            numExperts: 3,
            topK: 2,
            switchingDim: 512,
            temperature: 0.8
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
                    hiddenDim: this.config.headDim
                })
                .apply(outputs)

            outputs = this.ode.layers
                .AdaptiveMixtureOfExperts({
                    topK: this.config.topK,
                    switchingDim: this.config.switchingDim,
                    activation: 'swish',
                    temperature: this.config.temperature,
                    experts: this.createMLPExperts(outputs.shape)
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    createMLPExperts(inputShape) {
        return Array(this.config.numExperts)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'MultiLayerPerceptron',
                    inputShape,
                    hiddenDim: this.config.mlpDim,
                    activation: 'swish'
                })
            })
    }
}
