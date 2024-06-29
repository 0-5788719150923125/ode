import ODE from './model.v2.js'

/**
 * A maybe-sparse mixture of depths.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 6
        this.units = config.units || 256
        this.routerDim = config.routerDim || 64
        this.headDim = config.headDim || 512
        this.mlpDim = config.mlpDim || 1024
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: 'OriginalDesign/thrice'
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
            const attn = this.ode.layers.SelfAttention({
                projection: this.headDim
            })

            outputs = this.ode.layers
                .MixtureOfDepths({
                    layer: attn,
                    routerDim: this.routerDim,
                    activation: 'swish'
                })
                .apply(outputs)

            const ffd = this.ode.layers.GatedLinearMLP({
                innerDim: this.mlpDim,
                activation: 'gelu_new'
            })

            outputs = this.ode.layers
                .MixtureOfDepths({
                    layer: ffd,
                    routerDim: this.routerDim,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    createAttentionExperts(inputShape) {
        return Array(this.numExperts)
            .fill(0)
            .map((_, i) => {
                return this.ode.expert({
                    type: 'SelfAttention',
                    inputShape,
                    projection: this.headDim
                })
            })
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
                weightDecay: 0.1
            })
        ]
    }
}
