import ODE from './model.v2.js'

/**
 * A kinda-sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 8
        this.units = config.units || 32
        this.embeddings = config.embeddings || 256
        this.numExperts = config.numExperts || 23
        this.moeDim = config.moeDim || 128
        this.headDim = config.headDim || 1024
        this.mlpDim = config.mlpDim || 64
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
            outputDim: this.embeddings,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        outputs = this.ode.layers
            .dense({
                units: this.units,
                activation: 'mish'
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SelfAttention({
                    units: this.units,
                    hiddenDim: this.headDim
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SwarmOfExperts({
                    activation: 'mish',
                    hiddenDim: this.moeDim,
                    experts: this.createFeedforwardExperts(outputs.shape)
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.embeddings,
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
        return Array(this.numExperts)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'GatedLinearMLP',
                    inputShape,
                    hiddenDim: this.mlpDim,
                    activation: 'mish'
                })
            })
    }
}
