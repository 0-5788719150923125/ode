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

export default class ModelPrototype extends ModelBase {
    build() {
        super.build()

        this.model = tf.sequential()

        this.model.add(
            tf.layers.embedding({
                inputDim: this.vocab.length,
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
                units: this.vocab.length,
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
