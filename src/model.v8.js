import ODE from './model.v4.js'

/**
 * An experimental language model with limited memory footprint.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = 8
        this.units = 16
    }

    async defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.BinaryTokenizer()
    }

    // async defineTokenizer(config) {
    //     this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
    // }

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
                .SynthesizerAttention({
                    units: this.units,
                    blockSize: this.config.contextLength,
                    heads: 4
                })
                .apply(outputs)

            // outputs = this.ode.layers
            //     .LinearAttention({
            //         units: this.units
            //         // heads: 8,
            //         // topK: 4
            //     })
            //     .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    units: this.units,
                    innerDim: this.units * 4,
                    activation: 'mish'
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.learningRate = 0.00022
        this.optimizers[0].learningRate = this.learningRate
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: 0.001
            })
        ]
    }
}
