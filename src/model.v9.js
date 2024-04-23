import ODE from './model.v4.js'
import { randomValueFromArray } from './utils.js'

/**
 * For CPU-only peers.
 * @extends ODE
 */
export default class ObjectivelyDumbExample extends ODE {
    constructor(config) {
        super(config)
        this.units = 66
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
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

        outputs = this.ode.layers
            .SelfAttention({
                units: this.units,
                projection: this.units / 2
            })
            .apply(outputs)

        outputs = this.ode.layers
            .MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units / 2,
                activation: 'mish'
            })
            .apply(outputs)

        outputs = this.ode.layers
            .SelfAttention({
                units: this.units,
                projection: this.units / 3
            })
            .apply(outputs)

        outputs = this.ode.layers
            .MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units / 3,
                activation: 'mish'
            })
            .apply(outputs)

        // const numIterations = 9
        // for (let i = 0; i < numIterations; i++) {
        //     outputs = attn.apply(outputs)

        //     outputs = this.ode.layers.Bias({ l2: 0.1 }).apply(outputs)

        //     outputs = this.ode.layers
        //         .activation({
        //             activation: randomValueFromArray('mish', 'swish')
        //         })
        //         .apply(outputs)

        //     // outputs = this.ode.layers
        //     //     .ResidualConnection()
        //     //     .apply([outputs, activated])
        // }

        // outputs = this.ode.layers
        //     .MultiLayerPerceptron({
        //         units: this.units,
        //         innerDim: this.units * 4,
        //         activation: 'mish'
        //     })
        //     .apply(outputs)

        // outputs = this.ode.layers.Bias({ l2: 0.1 }).apply(outputs)
        // outputs = this.ode.layers.Bias({ l1: 0.1 }).apply(outputs)

        // outputs = this.ode.layers.Bias({ l1: 0.1 }).apply(outputs)
        // outputs = this.ode.layers.Bias({ l2: 0.1 }).apply(outputs)

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.learningRate = 0.0001
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
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
