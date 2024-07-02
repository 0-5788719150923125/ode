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
        this.numExperts = config.numExperts || 16
        this.moeDim = config.moeDim || 128
        this.headDim = config.headDim || 512
        this.mlpDim = config.mlpDim || 64
    }

    defineTokenizer() {
        super.defineTokenizer({
            model: 'OriginalDesign/twos'
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
                    units: this.units,
                    projection: this.headDim
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

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    createFeedforwardExperts(inputShape) {
        return Array(this.numExperts)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'GatedLinearMLP',
                    inputShape,
                    innerDim: this.mlpDim,
                    activation: 'mish'
                })
            })
    }
}
