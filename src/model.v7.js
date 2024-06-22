import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpenDoorExperiment extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 9
        this.embeddings = config.embeddings || 999
        this.units = config.units || 9
        this.heads = config.heads || 3
        this.queryRatio = config.queryRatio || 2
        this.headDim = config.headDim || 81
        this.mlpDim = config.mlpDim || 999
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer(config)
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.embeddings,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .PrincipalComponentAnalysis({ outputDim: this.units })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .GroupedQueryAttention({
                    heads: this.heads,
                    projection: this.headDim,
                    queryRatio: this.queryRatio
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'mish',
                    innerDim: this.mlpDim
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.minLearningRate = 0.00000001
        this.maxLearningRate = 0.00022
        const steps = 1000
        this.schedulers = [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.minLearningRate,
                this.maxLearningRate,
                steps
            )
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.maxLearningRate,
                weightDecay: 0.01
            })
        ]
    }
}
