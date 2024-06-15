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
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .RotaryPositionalEncoding({
                blockSize: this.config.contextLength
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .MixtureOfExperts({
                    experts: this.createAttentionExperts()
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    innerDim: this.units * 4,
                    activation: 'swish'
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
