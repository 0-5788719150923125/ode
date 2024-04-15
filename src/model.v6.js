import ODE from './model.v4.js'

/**
 * An experimental language model with limited memory footprint.
 * @extends ODE
 */
export default class OscillometricDecayedExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = 8
        this.units = 512
        this.maxDecisions = 9
        this.kernelSize = 3
        this.dilation = 3
        this.gamma = 3
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

        const encoding = this.ode.layers.RotaryPositionalEncoding({
            blockSize: this.config.contextLength
        })

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .Vectorrent({
                    maxDecisions: this.maxDecisions,
                    kernelSize: this.kernelSize,
                    dilation: this.dilation,
                    units: this.units,
                    gamma: this.gamma
                })
                .apply(outputs)

            // outputs = this.ode.layers
            //     .MultiLayerPerceptron({
            //         units: this.units,
            //         innerDim: this.units * 4,
            //         activation: 'mish'
            //     })
            //     .apply(outputs)

            outputs = this.ode.layers
                .PseudoQuantumState({
                    units: this.units,
                    qubits: 4,
                    strength: 0.8,
                    expansion: 4
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
