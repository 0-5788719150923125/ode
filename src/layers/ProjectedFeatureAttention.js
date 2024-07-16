import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 64
        this.headFeatures = config.headFeatures || 32
        this.queriesPerHead = config.queriesPerHead || 1
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []
        this.projectionKernels = []
        this.queryBiases = []
        this.keyBiases = []
        this.valueBiases = []
        this.projectionBiases = []

        const totalQueries = this.numHeads * this.queriesPerHead

        for (let i = 0; i < totalQueries; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            if (this.useBias)
                this.queryBiases.push(
                    this.addWeight(
                        `queryBias_${i}`,
                        [this.headDim],
                        'float32',
                        tf.initializers.zeros()
                    )
                )
        }

        for (let i = 0; i < this.numHeads; i++) {
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            if (this.useBias)
                this.keyBiases.push(
                    this.addWeight(
                        `keyBias_${i}`,
                        [this.headDim],
                        'float32',
                        tf.initializers.zeros()
                    )
                )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [inputDim, this.headFeatures],
                    'float32',
                    tf.initializers.orthogonal({ gain: 1.0 }),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            if (this.useBias)
                this.valueBiases.push(
                    this.addWeight(
                        `valueBias_${i}`,
                        [this.headFeatures],
                        'float32',
                        tf.initializers.zeros()
                    )
                )
            this.projectionKernels.push(
                this.addWeight(
                    `projectionKernel_${i}`,
                    [this.headDim, this.headFeatures],
                    'float32',
                    tf.initializers.orthogonal({ gain: 1.0 }),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            if (this.useBias)
                this.projectionBiases.push(
                    this.addWeight(
                        `projectionBias_${i}`,
                        [this.headFeatures],
                        'float32',
                        tf.initializers.zeros()
                    )
                )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headFeatures * this.numHeads * this.queriesPerHead, inputDim],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        if (this.useBias)
            this.outputBias = this.addWeight(
                'outputBias',
                [inputDim],
                'float32',
                tf.initializers.zeros()
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
                const K = this.ops.applyDense(
                    inputs,
                    this.keyKernels[i].read(),
                    this.useBias ? this.keyBiases[i].read() : null
                )
                const V = this.ops.applyDense(
                    inputs,
                    this.valueKernels[i].read(),
                    this.useBias ? this.valueBiases[i].read() : null
                )

                const KP = this.ops.applyDense(
                    K,
                    this.projectionKernels[i].read(),
                    this.useBias ? this.projectionBiases[i].read() : null
                )

                for (let j = 0; j < this.queriesPerHead; j++) {
                    const queryIndex = i * this.queriesPerHead + j
                    const Q = this.ops.applyDense(
                        inputs,
                        this.queryKernels[queryIndex].read(),
                        this.useBias
                            ? this.queryBiases[queryIndex].read()
                            : null
                    )

                    const QP = this.ops.applyDense(
                        Q,
                        this.projectionKernels[i].read(),
                        this.useBias ? this.projectionBiases[i].read() : null
                    )

                    let scores = tf.matMul(QP, KP, false, true)
                    scores = scores.div(tf.sqrt(tf.scalar(KP.shape[1])))

                    if (this.ALiBiLength) {
                        scores = this.ops.applyALiBi(
                            scores,
                            this.numHeads,
                            i,
                            seqLen,
                            this.ALiBiLength
                        )
                    }

                    const maskedScores = scores.add(mask)

                    const weights = maskedScores.softmax()

                    const output = tf.matMul(weights, V)

                    attentionOutputs.push(output)
                }
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)

            let outputs = this.ops.applyDense(
                concatenatedOutputs,
                this.outputKernel.read(),
                this.useBias ? this.outputBias.read() : null
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
            headFeatures: this.headFeatures,
            queriesPerHead: this.queriesPerHead
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureAttention)
