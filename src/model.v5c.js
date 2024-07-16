import ODE from './model.v2.js'

/**
 * A maybe-sparse mixture of depths.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.routerDim = config.routerDim || 1024
        this.headDim = config.headDim || 2048
        this.mlpDim = config.mlpDim || 1024
        this.capacity = config.capacity || 0.25
        this.learningRate = 0.00022
        this.minLearningRate = 0.00000001
        this.weightDecay = 0.001
        this.steps = 1024
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-consistent-v1'
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
                .MixtureOfDepths({
                    routerDim: this.routerDim,
                    activation: 'gelu_new',
                    capacity: this.capacity,
                    experts: [
                        this.ode.expert({
                            type: 'SelfAttention',
                            inputShape: outputs.shape,
                            projection: this.headDim
                        }),
                        this.ode.expert({
                            type: 'GatedLinearMLP',
                            inputShape: outputs.shape,
                            innerDim: this.mlpDim,
                            activation: 'gelu_new'
                        })
                    ]
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.minLearningRate,
                this.learningRate,
                this.steps
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
