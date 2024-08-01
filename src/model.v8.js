import ODE from './model.v2.js'

/**
 * Another transformer.
 * @extends ODE
 */
export default class OpportunisticDialogueEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.mlpDim = config.mlpDim || 768
        this.headDim = config.headDim || 256
        this.numHeads = config.numHeads || 3
        this.queriesPerHead = config.queriesPerHead || 2
        this.headFeatures = config.headFeatures || 64
        this.learningRate = 0.0001
        this.minLearningRate = 0.00000001
        this.weightDecay = 0.01
        this.cosineSteps = 4096
        this.ALiBiLength = 1024
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-balanced-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .ProjectedFeatureAttention({
                    numHeads: this.numHeads,
                    headDim: this.headDim,
                    headFeatures: this.headFeatures,
                    queriesPerHead: this.queriesPerHead,
                    ALiBiLength: this.ALiBiLength
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    innerDim: this.mlpDim,
                    activation: 'mish',
                    gateActivation: 'swish'
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

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
}
