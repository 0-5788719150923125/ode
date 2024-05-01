import ODE from './model.v3.js'

/**
 * A mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = 4
        this.heads = 8
        this.units = 256
        this.innerDim = this.units * 4
        this.epsilon = 1e-5
        this.alpha = 0.22
        this.topK = 2
        this.loadBalancing = 1.0
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
                .SimpleMixtureOfExperts({
                    experts: this.createAttentionExperts(),
                    topK: this.topK
                })
                .apply(outputs)

            outputs = this.ode.layers
                .CapsNet({
                    units: this.units,
                    innerDim: this.innerDim * 4,
                    epsilon: this.epsilon,
                    activation: 'swish'
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

    // defineSchedulers() {
    //     this.learningRate = 0.0001
    //     this.schedulers = [
    //         this.ode.schedulers.constantScheduler(this.learningRate)
    //     ]
    // }

    // defineOptimizers() {
    //     this.optimizers = [
    //         this.ode.optimizers.Lion({
    //             learningRate: this.learningRate,
    //             weightDecay: 0.1
    //         })
    //     ]
    // }

    createAttentionExperts() {
        return [
            this.ode.layers.SynthesizerAttention({
                units: this.units,
                blockSize: this.config.contextLength,
                heads: this.heads,
                epsilon: this.epsilon,
                activation: this.tf.leakyRelu,
                alpha: this.alpha
            }),
            this.ode.layers.SynthesizerAttention({
                units: this.units,
                blockSize: this.config.contextLength,
                heads: this.heads,
                epsilon: this.epsilon,
                activation: this.tf.selu
            }),
            this.ode.layers.SynthesizerAttention({
                units: this.units,
                blockSize: this.config.contextLength,
                heads: this.heads,
                epsilon: this.epsilon,
                activation: this.tf.tanh
            })
        ]
    }
}
