import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class GroupedQueryAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.heads = config.heads || 8
        this.headDim = config.headDim || 256
        this.queriesPerHead = config.queriesPerHead || 2
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.queryBiases = []
        this.keyKernels = []
        this.keyBiases = []
        this.valueKernels = []
        this.valueBiases = []

        for (let i = 0; i < this.heads; i++) {
            const queryKernels = []
            const queryBiases = []
            for (let j = 0; j < this.queriesPerHead; j++) {
                queryKernels.push(
                    this.addWeight(
                        `queryKernel-${i}-${j}`,
                        [units, this.headDim],
                        'float32',
                        tf.initializers.glorotUniform(),
                        tf.regularizers.l2({ l2: 0.01 })
                    )
                )
                queryBiases.push(
                    this.addWeight(
                        `queryBias-${i}-${j}`,
                        [this.headDim],
                        'float32',
                        tf.initializers.zeros(),
                        tf.regularizers.l2({ l2: 0.01 })
                    )
                )
            }
            this.queryKernels.push(queryKernels)
            this.queryBiases.push(queryBiases)

            this.keyKernels.push(
                this.addWeight(
                    `keyKernel-${i}`,
                    [units, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.keyBiases.push(
                this.addWeight(
                    `keyBiases-${i}`,
                    [this.headDim],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel-${i}`,
                    [units, units],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueBiases.push(
                this.addWeight(
                    `valueBiases-${i}`,
                    [units],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [units * this.heads * this.queriesPerHead, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputBias = this.addWeight(
            `outputBias`,
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const seqLen = inputs.shape[1]

            const attentionOutputs = []

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            for (let i = 0; i < this.heads; i++) {
                const K = this.ops.applyDense(
                    inputs,
                    this.keyKernels[i].read(),
                    this.keyBiases[i].read()
                )
                const V = this.ops.applyDense(
                    inputs,
                    this.valueKernels[i].read(),
                    this.valueBiases[i].read()
                )

                for (let j = 0; j < this.queriesPerHead; j++) {
                    const Q = this.ops.applyDense(
                        inputs,
                        this.queryKernels[i][j].read(),
                        this.queryBiases[i][j].read()
                    )

                    let scores = tf
                        .matMul(Q, K, false, true)
                        .div(tf.scalar(Math.sqrt(this.headDim)))

                    if (this.ALiBiLength) {
                        scores = this.ops.applyALiBi(
                            scores,
                            this.heads,
                            i,
                            seqLen,
                            this.ALiBiLength
                        )
                    }

                    scores = scores.add(mask)

                    let weights = scores.softmax()

                    weights = kwargs['training']
                        ? tf.dropout(weights, this.dropout)
                        : weights

                    const output = tf.matMul(weights, V)
                    attentionOutputs.push(output)
                }
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            let outputs = this.ops.applyDense(
                concatenatedOutputs,
                this.outputKernel.read(),
                this.outputBias.read()
            )

            outputs = this.ops.rmsNorm(outputs)

            outputs = tf.add(inputs, outputs)

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            headDim: this.headDim,
            heads: this.heads,
            queriesPerHead: this.queriesPerHead,
            dropout: this.dropout
        }
    }
}

tf.serialization.registerClass(GroupedQueryAttention)
