import ODE from './model.v0.js'

/**
 * An extremely simple text-generation RNN, used for next-token prediction. You
 * will see similar versions of this model everywhere, in tutorials on the Internet.
 * @extends ODE
 */
export default class OmnipresentDegenerateEntity extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 128
        this.labels = 'oneLabel'
        this.autoregressive = true
    }

    defineBuild() {
        const model = this.tf.sequential()

        model.add(
            this.tf.layers.embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units
            })
        )

        for (let i = 0; i < this.layers; i++)
            model.add(
                this.tf.layers.lstm({
                    units: this.units,
                    activation: 'tanh',
                    recurrentActivation: 'sigmoid',
                    returnSequences: i < this.layers - 1 // False for the last layer
                })
            )

        model.add(
            this.tf.layers.dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
        )

        return model
    }
}
