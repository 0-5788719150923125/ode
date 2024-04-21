import ODE from './model.v4.js'

/**
 * For CPU-only peers.
 * @extends ODE
 */
export default class ObjectivelyDumbExample extends ODE {
    constructor(config) {
        super(config)
        this.units = 256
    }

    // defineTokenizer(config) {
    //     this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
    // }
    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: 'OriginalDesign/word'
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

        if (rollDice())
            outputs = this.ode.layers
                .activation({ activation: 'swish' })
                .apply(outputs)

        outputs = this.ode.layers.Bias({ l1: 0.1 }).apply(outputs)

        if (rollDice())
            outputs = this.ode.layers
                .activation({ activation: 'mish' })
                .apply(outputs)

        outputs = this.ode.layers.Bias({ l2: 0.1 }).apply(outputs)

        if (rollDice())
            outputs = this.ode.layers
                .activation({ activation: 'softsign' })
                .apply(outputs)

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.learningRate = 1.0
        this.optimizers[0].learningRate = this.learningRate
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Prodigy({
                learningRate: this.learningRate,
                weightDecay: 0.1
            })
        ]
    }
}

function rollDice(threshold = 0.333) {
    if (Math.random() < threshold) return true
    else return false
}
