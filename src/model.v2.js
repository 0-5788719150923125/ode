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

        // Add the embedding layer as the first layer
        const inputs = tf.input({ shape: [null] })
        const embeddings = tf.layers
            .embedding({
                inputDim: this.vocab.length,
                outputDim: 256,
                // embeddingsInitializer: 'glorotUniform',
                // embeddingsConstraint: tf.constraints.maxNorm({
                //     maxValue: 0.2
                // }),
                // embeddingsRegularizer: tf.regularizers.l2(),
                maskZero: true
            })
            .apply(inputs)

        const layers = [512, 256, 128]
        let recurrentOutput = embeddings
        layers.forEach((size, i) => {
            const layer = tf.layers
                .gru({
                    units: size,
                    // dropout: 0,
                    // stateful: false,
                    activation: 'softsign',
                    // kernelInitializer: 'glorotUniform',
                    // kernelConstraint: tf.constraints.maxNorm({
                    //     axis: 0,
                    //     maxValue: 2.0
                    // }),
                    recurrentActivation: 'sigmoid',
                    // recurrentInitializer: 'orthogonal',
                    // recurrentConstraint: tf.constraints.maxNorm({
                    //     axis: 0,
                    //     maxValue: 2.0
                    // }),
                    returnSequences: i < layers.length - 1 // False for the last recurrent layer
                })
                .apply(recurrentOutput)

            recurrentOutput = layer
        })

        // Add the final dense layer
        const finalDense = tf.layers
            .dense({
                units: this.vocab.length,
                activation: 'linear'
            })
            .apply(recurrentOutput)

        this.model = tf.model({ inputs: inputs, outputs: finalDense })
    }
}
