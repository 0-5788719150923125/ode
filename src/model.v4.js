import ODE from './model.v2.js'

/**
 * A small transformer with synthesizer attention, GLU-based feedforward
 * networks, and sinusoidal positional encoding.
 * @extends ODE
 */
export default class OpportunisticDialogueEngine extends ODE {
    constructor(config) {
        const defaults = {
            layers: 6,
            units: 512,
            numHeads: 8,
            mlpDim: 2048,
            alpha: 0.22
        }
        super({ ...defaults, ...config })
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
                outputDim: this.config.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers.SinusoidalPositionalEncoding().apply(outputs)

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .SynthesizerAttention({
                    units: this.config.units,
                    blockSize: this.contextLength,
                    heads: this.config.numHeads,
                    alpha: this.config.alpha
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    hiddenDim: this.config.mlpDim,
                    activation: 'swish',
                    gateActivation: 'sigmoid'
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        return this.tf.model({ inputs, outputs })
    }
}
