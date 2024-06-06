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
        this.heads = config.heads || 2
        this.queryRatio = config.queryRatio || 3
        this.headDim = config.headDim || 64
        this.mlpDim = config.mlpDim || this.units * 4
        this.downsampling = {
            strategy: 'truncate',
            rate: 1.5
        }
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
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

        const autoencoder = this.ode.layers.Autoencoder({
            variational: true,
            downsampling: this.downsampling,
            innerDim: this.units * 4,
            bottleneck: this.units / 2,
            outputDim: this.units,
            encoderActivation: 'mish',
            decoderActivation: 'mish'
        })

        outputs = autoencoder.apply(outputs)

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
                    activation: 'selu',
                    innerDim: this.mlpDim
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
}
