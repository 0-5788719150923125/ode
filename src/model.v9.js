import ODE from './model.v8.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpenDoorExperiment extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 256
        this.embeddings = config.embeddings || 512
        this.heads = config.heads || 3
        this.queryRatio = config.queryRatio || 2
        this.headDim = config.headDim || 128
        this.mlpDim = config.mlpDim || 1024
        this.encoderDim = config.encoderDim || 1024
        this.bottleneck = config.bottleneck || 128
        this.beta = config.beta || 8.0
    }

    // defineTokenizer(config) {
    //     this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
    //         model: 'OriginalDesign/beast'
    //     })
    // }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer(config)
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
            .Autoencoder({
                variational: true,
                beta: this.beta,
                innerDim: this.encoderDim,
                bottleneck: this.bottleneck,
                outputDim: this.units,
                encoderActivation: 'mish',
                decoderActivation: 'mish'
            })
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
        const steps = 1024
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
