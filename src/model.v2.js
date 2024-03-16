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

class CausalAttentionLayer extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.dModel = config.dModel || 256 // Set default if dModel is not provided
    }

    build(inputShape) {
        // Initialize the necessary dense layers for internal transformations
        // These are correctly defined
        this.queryDense = tf.layers.dense({
            units: this.dModel,
            kernelInitializer: 'glorotUniform', // Use your preferred initializer
            useBias: false // Optionally disable bias if needed
        })
        this.keyDense = tf.layers.dense({
            units: this.dModel,
            kernelInitializer: 'glorotUniform',
            useBias: false
        })
        this.valueDense = tf.layers.dense({
            units: this.dModel,
            kernelInitializer: 'glorotUniform',
            useBias: false
        })

        // Ensuring internal layers are ready to be built with proper input shape
        const lastDimension = inputShape[inputShape.length - 1]
        this.queryDense.build([null, lastDimension])
        this.keyDense.build([null, lastDimension])
        this.valueDense.build([null, lastDimension])

        // Collecting weights from the internal layers manually if needed
        // This is not necessary for training but if you need to manually access these weights
        this._trainableWeights = [
            ...this.queryDense.trainableWeights,
            ...this.keyDense.trainableWeights,
            ...this.valueDense.trainableWeights
        ]

        super.build(inputShape) // Mark the layer as built
    }

    computeOutputShape(inputShape) {
        // Assume 'values' has the defining output shape
        return inputShape
    }

    call(inputs) {
        return tf.tidy(() => {
            const queries = this.queryDense.apply(inputs)
            const keys = this.keyDense.apply(inputs)
            const values = this.valueDense.apply(inputs)

            const keysTransposed = tf.transpose(keys, [0, 2, 1])

            let scores = tf.matMul(queries, keysTransposed)
            scores = tf.div(scores, tf.sqrt(tf.scalar(this.dModel)))

            // Creating a causal mask without using .triu()
            const seqLen = queries.shape[1]
            const mask = tf
                .tensor2d(
                    Array.from({ length: seqLen }, (_, i) =>
                        Array.from({ length: seqLen }, (_, j) =>
                            i >= j ? 0 : -1e9
                        )
                    )
                )
                .expandDims(0)
                .tile([queries.shape[0], 1, 1]) // Adjust the tiling to match batch size

            scores = tf.add(scores, mask)

            const attentionWeights = tf.softmax(scores, -1)
            const contextVector = tf.matMul(attentionWeights, values)

            return contextVector
        })
    }

    static get className() {
        return 'CausalAttentionLayer'
    }
}

tf.serialization.registerClass(CausalAttentionLayer)

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
                const attention = new CausalAttentionLayer({ dModel: 128 })
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
