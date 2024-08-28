import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

// By projecting features into a lower dimension, we can keep memory
// consumption at a constant, manageable level.
export default class ConstantSelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 256
        this.numFeatures = config.numFeatures || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputDim, this.hiddenDim],
            'float32',
            this.initializers.glorotUniform()
        )

        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputDim, this.hiddenDim],
            'float32',
            this.initializers.glorotUniform()
        )

        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputDim, inputDim],
            'float32',
            this.initializers.glorotUniform()
        )

        this.featureMatrix = this.addWeight(
            'featureMatrix',
            [this.hiddenDim, this.numFeatures],
            'float32',
            this.initializers.glorotUniform()
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.ops.applyDense(inputs, this.queryKernel.read())
            const K = this.ops.applyDense(inputs, this.keyKernel.read())
            const V = this.ops.applyDense(inputs, this.valueKernel.read())

            const Qp = this.ops.applyDense(Q, this.featureMatrix.read())
            const Kp = this.ops.applyDense(K, this.featureMatrix.read())

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf.matMul(Qp, Kp, false, true).add(mask)

            const weights = scores.div(tf.scalar(Math.sqrt(this.numFeatures)))

            const outputs = tf.matMul(weights, V)

            const normalized = this.ops.rmsNorm(outputs)

            return tf.add(inputs, normalized)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            numFeatures: this.numFeatures
        }
    }
}

tf.serialization.registerClass(ConstantSelfAttention)
