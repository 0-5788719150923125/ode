import ODE from './model.v0.js'

/**
 * An extremely simple text-generation RNN, used for next-token prediction. You
 * will see versions of this model everywhere, in tutorials on the Internet.
 * @extends ODE
 */
export default class OmnipotentDegenerateEntity extends ODE {
    constructor(config) {
        super(config)
        this.layers = 3
        this.units = 128
        this.config.labels = 'oneLabel'
    }

    defineBuild() {
        this.model = this.tf.sequential()

        this.model.add(
            this.tf.layers.embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                maskZero: true
            })
        )

        for (let i = 0; i < this.layers; i++)
            this.model.add(
                this.tf.layers.lstm({
                    units: this.units,
                    activation: 'tanh',
                    recurrentActivation: 'sigmoid',
                    returnSequences: i < this.layers - 1 // False for the last GRU layer
                })
            )

        this.model.add(
            this.tf.layers.dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
        )
    }
}
