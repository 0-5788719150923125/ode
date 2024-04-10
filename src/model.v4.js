import ODE from './model.v3.js'

/**
 * A small transformer with synthetic attention weights, GLU-based feedforward
 * networks, and rotary positional embeddings.
 * @extends ODE
 */
export default class OpportunisticDialogueEncoder extends ODE {
    constructor(config) {
        super(config)
        this.layers = 6
        this.heads = 8
        this.units = 512
        this.innerDim = this.units * 4
        this.epsilon = 1e-5
        this.alpha = 0.22
    }

    async defineTokenizer() {
        await super.defineTokenizer({
            model: 'OriginalDesign/frame'
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
                .SynthesizerAttention({
                    units: this.units,
                    blockSize: this.config.contextLength,
                    heads: this.heads,
                    epsilon: this.epsilon,
                    alpha: this.alpha
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearUnit({
                    units: this.units,
                    innerDim: this.innerDim,
                    heads: this.heads,
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
}
