import ODE from './model.v3.js'

/**
 * An experimental, deterministic language model with next to 0 trainable parameters.
 * @extends ODE
 */
export default class OscillatingDecayedExponent extends ODE {
    constructor(config) {
        super(config)
        this.layers = 9
        this.units = 64
        this.routingIterations = 9
        this.kernelSize = 3
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

        const embeddings = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.ode.layers.Range().apply(inputs)

        const encoding = this.ode.layers
            .embedding({
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotNormal'
            })
            .apply(range)

        let outputs = this.ode.layers.add().apply([embeddings, encoding])

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .Vectorrent({
                    routingIterations: this.routingIterations,
                    kernelSize: this.kernelSize,
                    units: this.units
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
