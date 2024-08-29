import ODE from './model.v0.js'

/**
 * An extremely simple text-generation RNN, used for next-token prediction. You
 * will see similar versions of this model everywhere, in tutorials on the Internet.
 * @extends ODE
 */
export default class OmnipresentDegenerateEntity extends ODE {
    constructor(config) {
        const defaults = {
            layers: 3,
            units: 128,
            learningRate: 1e-4,
            weightDecay: 1e-5
        }
        super({ ...defaults, ...config })
        this.labels = 'oneLabel'
        this.autoregressive = true
    }

    defineBuild() {
        const model = this.tf.sequential()

        model.add(
            this.tf.layers.embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.config.units
            })
        )

        for (let i = 0; i < this.config.layers; i++)
            model.add(
                this.tf.layers.lstm({
                    units: this.config.units,
                    activation: 'tanh',
                    recurrentActivation: 'sigmoid',
                    returnSequences: i < this.config.layers - 1 // False for the last layer
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
