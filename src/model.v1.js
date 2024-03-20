import ModelBase from './model.v0.js'

/**
 * An extremely simple text-generation RNN, used for next-token prediction. You
 * will see versions of this model everywhere, in tutorials on the Internet.
 * @extends ModelBase
 */
export default class ModelPrototype extends ModelBase {
    build() {
        super.build()

        this.model = tf.sequential()

        this.model.add(
            tf.layers.embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.config.embeddingDimensions,
                maskZero: true
            })
        )

        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        activation: 'tanh',
                        recurrentActivation: 'sigmoid',
                        returnSequences: i < this.config.layout.length - 1 // False for the last GRU layer
                    }),
                    mergeMode: 'concat'
                })
            )
        })

        this.model.add(
            tf.layers.dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
        )

        this.lossFunctions = [tf.losses.softmaxCrossEntropy]
        this.model.compile({
            optimizer: tf.train.rmsprop(
                this.config.learningRate || 1e-2,
                this.config.decay || 0,
                this.config.momentum || 0,
                this.config.epsilon || 1e-8
            ),
            loss: this.lossFunctions
        })
    }
}
