import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 64
        this.headFeatures = config.headFeatures || 32
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []
        this.projectionKernels = []

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
                    [inputDim, this.headFeatures],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
            this.projectionKernels.push(
                this.addWeight(
                    `projectionKernel_${i}`,
                    [this.headDim, this.headFeatures],
                    'float32',
                    tf.initializers.glorotUniform()
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headFeatures * this.numHeads, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const seqLen = inputs.shape[1]
            const mask = tf.linalg
                .bandPart(tf.ones([seqLen, seqLen]), 0, -1)
                .sub(tf.eye(seqLen))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.numHeads; i++) {
                const Q = this.applyDense(inputs, this.queryKernels[i].read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                const QP = this.applyDense(Q, this.projectionKernels[i].read())
                const KP = this.applyDense(K, this.projectionKernels[i].read())

                const QK = tf.matMul(QP, KP, false, true)
                const scores = QK.div(tf.sqrt(tf.scalar(KP.shape[1]))).add(mask)

                const weights = scores.softmax()

                const output = tf.matMul(weights, V)

                attentionOutputs.push(output)
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

    getWeights() {
        const weights = []

        for (let i = 0; i < this.numHeads; i++) {
            weights.push(this.queryKernels[i].read())
            weights.push(this.keyKernels[i].read())
            weights.push(this.valueKernels[i].read())
            weights.push(this.projectionKernels[i].read())
        }

        weights.push(this.outputKernel.read())

        return weights
    }

    setWeights(weights) {
        let index = 0

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels[i].write(weights[index++])
            this.keyKernels[i].write(weights[index++])
            this.valueKernels[i].write(weights[index++])
            this.projectionKernels[i].write(weights[index++])
        }

        this.outputKernel.write(weights[index])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numHeads: this.numHeads,
            headDim: this.headDim,
            headFeatures: this.headFeatures
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureAttention)
