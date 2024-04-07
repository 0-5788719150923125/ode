import ODE from './model.v3.js'

/**
 * An experimental language model.
 * @extends ODE
 */
export default class OscillatingDepthwiseEntanglement extends ODE {
    constructor(config) {
        super(config)
        this.layers = 23
        this.units = 23
        this.routingIterations = 27
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

        const tokenEmbeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.ode.layers.Range().apply(inputs)

        const positionalEmbeddings = this.tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotNormal'
            })
            .apply(range)

        let outputs = this.tf.layers
            .add()
            .apply([tokenEmbeddings, positionalEmbeddings])

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .Vectorrent({ routingIterations: 9 })
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
