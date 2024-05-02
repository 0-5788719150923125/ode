import ODE from './model.v3.js'

/**
 * A mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = 6
        this.units = 64
        this.topK = 2
    }

    defineTokenizer() {
        super.defineTokenizer({
            model: 'OriginalDesign/word'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units * 4,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .RotaryPositionalEncoding({
                blockSize: this.config.contextLength
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.units
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    experts: this.createAttentionExperts(),
                    units: this.units,
                    topK: this.topK
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    experts: this.createFeedforwardExperts(),
                    units: this.units,
                    topK: this.topK
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.learningRate = 0.001
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    createAttentionExperts() {
        return [
            this.ode.layers.MultiQueryAttention({
                units: this.units,
                projection: this.units * 4,
                queries: 4
            }),
            this.ode.layers.MultiQueryAttention({
                units: this.units,
                projection: this.units * 4,
                queries: 4
            }),
            this.ode.layers.GroupedQueryAttention({
                units: this.units,
                projection: this.units * 4,
                groups: 4
            }),
            this.ode.layers.GroupedQueryAttention({
                units: this.units,
                projection: this.units * 4,
                groups: 4
            }),
            this.ode.layers.MultiHeadAttention({
                units: this.units,
                heads: 4
            }),
            this.ode.layers.MultiHeadAttention({
                units: this.units,
                heads: 4
            })
        ]
    }

    createFeedforwardExperts() {
        return [
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units * 4,
                activation: 'mish'
            }),
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units * 4,
                activation: 'swish'
            }),
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units * 4,
                activation: 'aptx'
            })
        ]
    }
}
