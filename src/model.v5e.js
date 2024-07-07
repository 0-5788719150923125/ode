import ODE from './model.v2.js'

/**
 * A Soft Merging of Experts with Adaptive Routing.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 4
        this.units = config.units || 256
        this.embeddings = config.embeddings || 256
        this.numExperts = config.numExperts || 3
        this.moeDim = config.moeDim || 512
        this.headDim = config.headDim || 512
        this.numHeads = config.numHeads || 4
        this.headFeatures = config.headFeatures || 128
        this.mlpDim = config.mlpDim || 1024
        this.learningRate = 1e-4
        this.weightDecay = 0.001
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-8000-consistent-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.embeddings,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        // outputs = this.ode.layers
        //     .VarianceThreshold({
        //         outputDim: this.units
        //     })
        //     .apply(outputs)

        // outputs = this.ode.layers
        //     .IndependentComponentAnalysis({
        //         outputDim: this.units
        //     })
        //     .apply(outputs)

        // outputs = this.ode.layers
        //     .dense({
        //         units: this.units
        //     })
        //     .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .ProjectedFeatureAttention({
                    numHeads: this.numHeads,
                    headDim: this.headDim,
                    numFeatures: this.headFeatures
                })
                .apply(outputs)

            // outputs = this.ode.layers
            //     .SelfAttention({
            //         hiddenDim: this.headDim
            //     })
            //     .apply(outputs)

            outputs = this.ode.layers
                .SMEAR({
                    activation: 'mish',
                    hiddenDim: this.moeDim,
                    experts: this.createFeedforwardExperts(outputs.shape)
                })
                .apply(outputs)
            // outputs = this.ode.layers
            //     .GatedLinearMLP({
            //         innerDim: this.mlpDim,
            //         activation: 'mish'
            //     })
            //     .apply(outputs)
        }

        // outputs = this.ode.layers
        //     .dense({
        //         units: this.embeddings,
        //         activation: 'mish'
        //     })
        //     .apply(outputs)

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay
            })
        ]
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    createFeedforwardExperts(inputShape) {
        // We add 1 extra expert, since the first one is a weighted average of all other experts.
        return Array(this.numExperts + 1)
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
