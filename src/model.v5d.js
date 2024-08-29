import ODE from './model.v2.js'

/**
 * A kinda-sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        const defaults = {
            layers: 8,
            units: 32,
            embeddings: 256,
            numExperts: 23,
            moeDim: 128,
            headDim: 1024,
            mlpDim: 64
        }
        super({ ...defaults, ...config })
    }

    defineTokenizer() {
        super.defineTokenizer({
            model: 'OriginalDesign/thrice'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.embedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.config.embeddings,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        outputs = this.ode.layers
            .dense({
                units: this.config.units,
                activation: 'mish'
            })
            .apply(outputs)

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .SelfAttention({
                    units: this.config.units,
                    hiddenDim: this.config.headDim
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SwarmOfExperts({
                    activation: 'mish',
                    hiddenDim: this.config.moeDim,
                    experts: this.createFeedforwardExperts(outputs.shape)
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.config.embeddings,
                activation: 'mish'
            })
            .apply(outputs)

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    createFeedforwardExperts(inputShape) {
        return Array(this.config.numExperts)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'GatedLinearMLP',
                    inputShape,
                    hiddenDim: this.config.mlpDim,
                    activation: 'mish'
                })
            })
    }
}
