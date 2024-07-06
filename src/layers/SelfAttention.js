import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.projection = config.projection || 256
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernel = this.addWeight(
            `queryKernel`,
            [inputDim, this.projection],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.keyKernel = this.addWeight(
            `keyKernel`,
            [inputDim, this.projection],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.valueKernel = this.addWeight(
            `valueKernel`,
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.applyDense(inputs, this.queryKernel.read())
            const K = this.applyDense(inputs, this.keyKernel.read())
            const V = this.applyDense(inputs, this.valueKernel.read())

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf
                .matMul(Q, K, false, true)
                .div(tf.scalar(this.projection).sqrt())
                .add(mask)

            const weights = scores.softmax()

            let outputs = tf.matMul(weights, V)

            outputs = this.ops.rmsNorm(outputs)

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
            projection: this.projection
        }
    }
}

tf.serialization.registerClass(SelfAttention)
