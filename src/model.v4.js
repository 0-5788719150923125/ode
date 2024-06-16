import ODE from './model.v1.js'

/**
 * A small transformer with synthesizer attention, GLU-based feedforward
 * networks, and sinusoidal positional encoding.
 * @extends ODE
 */
export default class OpportunisticDialogueEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 6
        this.heads = config.heads || 8
        this.units = config.units || 512
        this.innerDim = config.innerDim || this.units * 4
        this.epsilon = config.epsilon || false
        this.alpha = config.alpha || 0.22
    }

    defineTokenizer() {
        super.defineTokenizer({
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

        outputs = this.ode.layers.SinusoidalPositionalEncoding().apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SynthesizerAttention({
                    units: this.units,
                    blockSize: this.contextLength,
                    heads: this.heads,
                    epsilon: this.epsilon,
                    alpha: this.alpha
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    innerDim: this.innerDim,
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
