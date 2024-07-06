import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// A chunk-based approach leads to O(n * chunk_size) memory, which is
// linear if chunk size is fixed.
// TODO: sliding window and heirarchical versions of this
export default class ConstantSelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units || 128
        this.projection = config.projection || 256
        this.chunkSize = config.chunkSize || 64
    }

    build(inputShape) {
        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputShape[inputShape.length - 1], this.projection],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputShape[inputShape.length - 1], this.projection],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputShape[inputShape.length - 1], this.units],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.applyDense(inputs, this.queryKernel.read())
            const K = this.applyDense(inputs, this.keyKernel.read())
            const V = this.applyDense(inputs, this.valueKernel.read())

            const numChunks = Math.ceil(inputs.shape[1] / this.chunkSize)
            const chunkOutputs = []

            for (let i = 0; i < numChunks; i++) {
                const start = i * this.chunkSize
                const end = Math.min((i + 1) * this.chunkSize, inputs.shape[1])

                const chunkQ = Q.slice(
                    [0, start, 0],
                    [Q.shape[0], end - start, Q.shape[2]]
                )
                const chunkK = K.slice(
                    [0, start, 0],
                    [K.shape[0], end - start, K.shape[2]]
                )
                const chunkV = V.slice(
                    [0, start, 0],
                    [V.shape[0], end - start, V.shape[2]]
                )

                const mask = tf.linalg
                    .bandPart(
                        tf.ones([chunkQ.shape[1], chunkQ.shape[1]]),
                        0,
                        -1
                    )
                    .sub(tf.eye(chunkQ.shape[1]))
                    .mul(tf.scalar(-1e9))

                const scores = tf
                    .matMul(chunkQ, chunkK, false, true)
                    .div(tf.scalar(this.projection).sqrt())
                    .add(mask)

                const weights = scores.softmax()

                const chunkOutput = tf.matMul(weights, chunkV)
                chunkOutputs.push(chunkOutput)
            }

            const outputs = tf.concat(chunkOutputs, 1)

            return tf.add(inputs, outputs)
        })
    }

    getWeights() {
        return [
            this.queryKernel.read(),
            this.keyKernel.read(),
            this.valueKernel.read()
        ]
    }

    setWeights(weights) {
        this.queryKernel.write(weights[0])
        this.keyKernel.write(weights[1])
        this.valueKernel.write(weights[2])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            projection: this.projection,
            chunkSize: this.chunkSize
        }
    }
}

tf.serialization.registerClass(ConstantSelfAttention)
