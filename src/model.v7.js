import ODE from './model.v3.js'

/**
 * A mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = 3
        this.units = 256
        this.experts = 3
        this.moeDim = 128
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

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform'
        })

        let outputs = embeddings.apply(inputs)

        outputs = this.ode.layers
            .RotaryPositionalEncoding({
                blockSize: this.contextLength
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            const experts = this.createAttentionExperts()
            const expertOutputs = experts.map((expert) => expert.apply(outputs))

            outputs = this.ode.layers
                .MixtureOfExperts({
                    numExperts: experts.length,
                    hiddenDim: this.moeDim,
                    activation: 'swish'
                })
                .apply([outputs, ...expertOutputs])

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    innerDim: this.units * 4,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

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
            this.ode.layers.SelfAttention({
                projection: this.units * 4
            }),
            this.ode.layers.SelfAttention({
                projection: this.units * 4
            }),
            this.ode.layers.SelfAttention({
                projection: this.units * 4
            })
        ]
    }
}
