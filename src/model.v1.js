import * as tfjs from '@tensorflow/tfjs'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
        tf = await import('@tensorflow/tfjs-node-gpu')
    }
})()

import '@tensorflow/tfjs-backend-wasm'
import '@tensorflow/tfjs-backend-webgpu'
import '@tensorflow/tfjs-backend-webgl'
import ModelBase from './model.v0.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    build() {
        super.build()

        this.model = tf.sequential()

        // Add the embedding layer as the first layer
        this.model.add(
            tf.layers.embedding({
                inputDim: this.vocab.length, // Should match size of the vocabulary
                outputDim: this.config.embeddingDimensions, // Dimension of the embedding vectors
                maskZero: true
            })
        )

        // Add GRU layers
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

        // Add the final dense layer
        this.model.add(
            tf.layers.dense({
                units: this.vocab.length,
                activation: 'linear'
            })
        )

        // Compile the model
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
