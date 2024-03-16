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
import { CausalAttentionLayer } from './layers.js'

export default class OmniscientDeterministicEngine extends ModelBase {
    build() {
        super.build()

        // Add the embedding layer as the first layer
        const inputs = tf.input({ shape: [null] })
        const embeddings = tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: 64,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        const layers = [128, 128, 128]
        let recurrentOutput = embeddings
        layers.forEach((size, i) => {
            const intermediateLayer = i < layers.length - 1
            const layer = tf.layers
                .gru({
                    units: size,
                    activation: 'softsign',
                    kernelInitializer: 'glorotUniform',
                    recurrentActivation: 'sigmoid',
                    recurrentInitializer: 'orthogonal',
                    returnSequences: intermediateLayer
                })
                .apply(recurrentOutput)

            recurrentOutput = layer

            if (intermediateLayer) {
                const attention = new CausalAttentionLayer({ units: 128 })
                recurrentOutput = attention.apply(recurrentOutput)
            }
        })

        // Add the final dense layer
        const finalDense = tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(recurrentOutput)

        this.model = tf.model({ inputs: inputs, outputs: finalDense })
    }
}
