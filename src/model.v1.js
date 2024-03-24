import ModelBase from './model.v0.js'

/**
 * An extremely simple text-generation RNN, used for next-token prediction. You
 * will see versions of this model everywhere, in tutorials on the Internet.
 * @extends ModelBase
 */
export default class ModelPrototype extends ModelBase {
    constructor(config) {
        super(config)
        this.layout = [128, 128]
        this.units = 256
    }

    build() {
        this.model = this.tf.sequential()

        this.model.add(
            this.tf.layers.embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                maskZero: true
            })
        )

        this.layout.forEach((units, i) => {
            this.model.add(
                this.tf.layers.bidirectional({
                    layer: this.tf.layers.gru({
                        units,
                        activation: 'tanh',
                        recurrentActivation: 'sigmoid',
                        returnSequences: i < this.layout.length - 1 // False for the last GRU layer
                    }),
                    mergeMode: 'concat'
                })
            )
        })

        this.model.add(
            this.tf.layers.dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
        )
    }
}
