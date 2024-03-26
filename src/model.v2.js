import ModelBase from './model.v0.js'

/**
 * An attempt to implement causal attention in a GRU-based RNN. It didn't work very well.
 * @extends ModelBase
 */
export default class OmnipresentDegenerateEntity extends ModelBase {
    constructor(config) {
        super(config)
        this.units = 128
        this.layers = new Array(5).fill(this.units)
        this.epsilon = 1e-3
    }

    defineBuild() {
        const inputs = this.tf.input({ shape: [null] })
        let outputs = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units * 2,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        this.layers.forEach((units, i) => {
            const notFirstLayer = i !== 0
            const notLastLayer = i !== layers.length - 1
            const recurrent = this.tf.layers
                .gru({
                    units,
                    activation: 'softsign',
                    kernelInitializer: 'glorotUniform',
                    recurrentActivation: 'sigmoid',
                    recurrentInitializer: 'orthogonal',
                    returnSequences: notLastLayer
                })
                .apply(outputs)

            if (notFirstLayer && notLastLayer) {
                outputs = this.ode.layers
                    .ResidualConnection()
                    .apply([outputs, recurrent])
            } else {
                outputs = recurrent
            }

            outputs = this.tf.layers
                .layerNormalization({
                    epsilon: this.epsilon
                })
                .apply(outputs)

            if (notLastLayer) {
                outputs = this.ode.layers
                    .CausalAttentionLayer({
                        units,
                        kernelInitializer: 'glorotUniform'
                    })
                    .apply(outputs)
            }
        })

        outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}
