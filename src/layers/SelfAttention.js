import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 256
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernel = this.addWeight(
            `queryKernel`,
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        this.keyKernel = this.addWeight(
            `keyKernel`,
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        this.valueKernel = this.addWeight(
            `valueKernel`,
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.ops.applyDense(inputs, this.queryKernel.read())
            const K = this.ops.applyDense(inputs, this.keyKernel.read())
            const V = this.ops.applyDense(inputs, this.valueKernel.read())

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf
                .matMul(Q, K, false, true)
                .div(tf.scalar(this.hiddenDim).sqrt())
                .add(mask)

            const weights = scores.softmax()

            let outputs = tf.matMul(weights, V)

            outputs = this.ops.rmsNorm(outputs)

            return tf.add(inputs, outputs)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim
        }
    }
}

tf.serialization.registerClass(SelfAttention)
