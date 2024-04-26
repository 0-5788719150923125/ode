import ODE from './model.v4.js'

/**
 * A state space model.
 * @extends ODE
 */
export default class OrthogonalDepthwiseEntanglement extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 1
        this.units = config.units || 512
        this.innerDim = config.innerDim || this.units * 4
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

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .StatePlacement({
                    units: this.units,
                    innerDim: this.innerDim,
                    returnSequences: true
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
