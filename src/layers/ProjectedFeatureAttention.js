import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headDim * this.numHeads, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const attentionOutputs = []

            for (let i = 0; i < this.numHeads; i++) {
                const Q = this.applyDense(inputs, this.queryKernels[i].read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                const QF = this.simpleFeatureMap(Q)
                const KF = this.simpleFeatureMap(K)

                const attention = this.causalAttention(KF, QF, V)

                attentionOutputs.push(attention)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)

            let outputs = this.applyDense(
                concatenatedOutputs,
                this.outputKernel.read()
            )

            outputs = this.ops.rmsNorm(outputs)

            outputs = tf.add(inputs, outputs)

            return outputs
        })
    }

    causalAttention(KF, QF, V) {
        const ref_v = tf.ones([...V.shape.slice(0, -1), 1])
        const Gps = tf.mul(tf.expandDims(KF, 2), tf.expandDims(V, 1))
        const Grenorm = tf.mul(tf.expandDims(KF, 2), tf.expandDims(ref_v, 1))

        const attRaw = tf.sum(tf.mul(Gps, tf.expandDims(QF, 1)), -1)
        const attNorm = tf.sum(tf.mul(Grenorm, tf.expandDims(QF, 1)), -1)

        const attRawCumsum = tf.cumsum(attRaw, 2)
        const attNormCumsum = tf.cumsum(attNorm, 2)

        const att = tf.div(attRawCumsum, attNormCumsum)

        const attendedValues = tf.matMul(att, V)

        return attendedValues
    }

    simpleFeatureMap(x) {
        return tf.relu(x)
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numHeads: this.numHeads,
            headDim: this.headDim
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureAttention)
