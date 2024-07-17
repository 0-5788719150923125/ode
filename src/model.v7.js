import ODE from './model.v2.js'

/**
 * A Frankenstein.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 5
        this.units = config.units || 128
        this.embeddings = config.embeddings || 512
        this.autoencoderDim = config.autoencoderDim || 768
        this.bottleneck = config.bottleneck || 96
        this.beta = config.beta || 1.1
        this.numExperts = config.numExperts || 3
        this.routerDim = config.routerDim || 768
        this.numHeads = config.numHeads || 3
        this.queriesPerHead = config.queriesPerHead | 3
        this.headDim = config.headDim || 96
        this.headFeatures = config.headFeatures || 64
        this.mlpDim = config.mlpDim || 512
        this.learningRate = 0.0001
        this.minLearningRate = 0.00000001
        this.weightDecay = 0.01
        this.cosineSteps = 2048
        this.ALiBiLength = 1024
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

        let outputs = embeddings.apply(inputs)

        outputs = this.ode.layers
            .StaticMatrixReduction({
                outputDim: this.units
            })
            .apply(outputs)

        // outputs = this.ode.layers
        //     .Autoencoder({
        //         innerDim: this.innerDim,
        //         bottleneck: this.bottleneck,
        //         outputDim: this.units,
        //         encoderActivation: 'tanh',
        //         decoderActivation: 'sigmoid',
        //         variational: true,
        //         beta: this.beta
        //     })
        //     .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .ProjectedFeatureAttention({
                    numHeads: this.numHeads,
                    queriesPerHead: this.queriesPerHead,
                    headDim: this.headDim,
                    headFeatures: this.headFeatures,
                    ALiBiLength: this.ALiBiLength
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SoftMergingOfExperts({
                    activation: 'mish',
                    hiddenDim: this.routerDim,
                    experts: this.createMLPExperts(outputs.shape)
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.embeddings
            })
            .apply(outputs)

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.minLearningRate,
                this.learningRate,
                this.cosineSteps
            )
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay
            })
        ]
    }

    createMLPExperts(inputShape) {
        // We add 1 extra expert, since the first one is an in-place, weighted average of all other experts.
        return Array(this.numExperts + 1)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'GatedLinearMLP',
                    inputShape,
                    innerDim: this.mlpDim,
                    activation: 'mish',
                    gateActivation: 'swish' // implements SwiGLU
                })
            })
    }
}
