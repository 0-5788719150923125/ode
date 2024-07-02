import ODE from './model.v2.js'

/**
 * A Soft Merging of Experts with Adaptive Routing.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 6
        this.units = config.units || 128
        this.embeddings = config.embeddings || 256
        this.numExperts = config.numExperts || 8
        this.moeDim = config.moeDim || 256
        this.headDim = config.headDim || 1024
        this.numFeatures = config.numFeatures || 256
        this.mlpDim = config.mlpDim || 512
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonsterTokenizer({
            model: 'englishcode-4096-clean-v1'
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
                .RandomFeatureAttention({
                    hiddenDim: this.headDim,
                    numFeatures: this.numFeatures,
                    numHeads: 1
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SMEARMoE({
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
