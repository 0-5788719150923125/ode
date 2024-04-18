import ODE from './model.v4.js'

/**
 * An experimental language model with limited memory footprint.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.units = 512
    }

    async defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.embedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        outputs = this.ode.layers
            .Collideoscope({
                units: this.units,
                qubits: 256,
                iterations: 9
            })
            .apply(outputs)

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        const learningRate = 0.001
        this.optimizers[0].learningRate = learningRate
        this.schedulers = [this.ode.schedulers.constantScheduler(learningRate)]
    }
}
