import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OutliarDerivativeExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 111
        this.heads = config.heads || 3
        this.queryRatio = config.queryRatio || 2
        this.headDim = config.headDim || 111
        this.mlpDim = config.mlpDim || 999
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: 'OriginalDesign/thrice'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const mlp = this.ode.layers.MeltingMLP({
            activation: 'mish',
            units: this.units * this.layers,
            innerDim: this.mlpDim
        })

        for (let i = 1; i <= this.layers; i++) {
            outputs = this.ode.layers
                .GroupedQueryAttention({
                    units: this.units * i,
                    heads: this.heads * i,
                    projection: this.headDim * i,
                    queryRatio: this.queryRatio
                })
                .apply(outputs)
            outputs = this.ode.layers
                .dense({
                    units: this.units * i
                })
                .apply(outputs)
            outputs = mlp.apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.minLearningRate = 0.00000000333
        this.maxLearningRate = 0.000333
        const steps = 1111
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
