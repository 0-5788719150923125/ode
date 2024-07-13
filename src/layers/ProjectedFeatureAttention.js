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
                const Q = this.ops.applyDense(
                    inputs,
                    this.queryKernels[i].read()
                )
                const K = this.ops.applyDense(inputs, this.keyKernels[i].read())
                const V = this.ops.applyDense(
                    inputs,
                    this.valueKernels[i].read()
                )

                const QP = this.ops.applyDense(
                    Q,
                    this.projectionKernels[i].read()
                )
                const KP = this.ops.applyDense(
                    K,
                    this.projectionKernels[i].read()
                )

                let scores = tf.matMul(QP, KP, false, true)
                scores = scores.div(tf.sqrt(tf.scalar(KP.shape[1])))

                if (this.useALiBi) {
                    scores = this.ops.applyALiBi(
                        scores,
                        this.numHeads,
                        i,
                        seqLen,
                        2048
                    )
                }

                const maskedScores = scores.add(mask)

                const weights = maskedScores.softmax()

                const output = tf.matMul(weights, V)

                attentionOutputs.push(output)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)

            let outputs = this.ops.applyDense(
                concatenatedOutputs,
                this.outputKernel.read()
            )

            outputs = this.ops.rmsNorm(outputs)

            outputs = tf.add(inputs, outputs)

            return outputs
        })
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
