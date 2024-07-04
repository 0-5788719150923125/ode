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
        this.numHeads = config.numHeads || 2
        this.numFeatures = config.numFeatures || 256
        this.mlpDim = config.mlpDim || 512
        this.learningRate = 1e-4
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
        //     .IndependentComponentAnalysis({
        //         outputDim: this.units
        //     })
        //     .apply(outputs)
        outputs = this.ode.layers
            .RandomProjectionFeatureReduction({
                outputDim: this.units
            })
            .apply(outputs)

        // outputs = this.ode.layers
        //     .dense({
        //         units: this.units
        //     })
        //     .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .RandomFeatureAttention({
                    hiddenDim: this.headDim,
                    numFeatures: this.numFeatures,
                    numHeads: this.numHeads
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

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: 0.001
            })
        ]
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
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
