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
                inputDim: this.vocab.length, // Should match size of the vocabulary
                outputDim: this.config.embeddingDimensions, // Dimension of the embedding vectors
                embeddingsInitializer: 'glorotUniform',
                embeddingsConstraint: tf.constraints.maxNorm({
                    maxValue: 0.2
                }),
                embeddingsRegularizer: tf.regularizers.l2(),
                maskZero: true
            })
            .apply(inputs)

        // Apply dropout on the embeddings layer
        const dropout1 = tf.layers.dropout({ rate: 0.1 }).apply(embeddings)

        // Add recurrent layers
        let previousLayerOutput = dropout1
        this.config.layout.forEach((units, i) => {
            const bidirectional = tf.layers
                .bidirectional({
                    layer: tf.layers.lstm({
                        units: units,
                        dropout: 0.1,
                        stateful: false,
                        activation: 'softsign',
                        kernelInitializer: 'glorotUniform',
                        kernelConstraint: tf.constraints.maxNorm({
                            axis: 0,
                            maxValue: 2.0
                        }),
                        recurrentActivation: 'sigmoid',
                        recurrentInitializer: 'orthogonal',
                        recurrentConstraint: tf.constraints.maxNorm({
                            axis: 0,
                            maxValue: 2.0
                        }),
                        returnSequences: i < this.config.layout.length - 1 // False for the last recurrent layer
                    }),
                    mergeMode: 'concat'
                })
                .apply(previousLayerOutput)

            const layerNorm = tf.layers
                .layerNormalization({
                    epsilon: 1e-3
                })
                .apply(bidirectional)
            previousLayerOutput = layerNorm
        })

        const dense1 = tf.layers
            .dense({
                units: 4096,
                activation: 'linear',
                useBias: true,
                kernelInitializer: 'randomUniform'
            })
            .apply(previousLayerOutput)

        // Apply dropout on the embeddings layer
        const dropout2 = tf.layers.dropout({ rate: 0.1 }).apply(dense1)

        // Add the final dense layer
        const finalDense = tf.layers
            .dense({
                units: this.vocab.length,
                activation: 'linear'
            })
            .apply(dropout2)

        this.model = tf.model({ inputs: inputs, outputs: finalDense })
    }
}
