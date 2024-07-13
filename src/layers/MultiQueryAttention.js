import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class MultiQueryAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.projection = config.projection || 256
        this.queries = config.queries || 8
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.queryBiases = []
        for (let i = 0; i < this.queries; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.queryBiases.push(
                this.addWeight(
                    `queryBiases${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
        }
        this.keyKernel = this.addWeight(
            'keyKernel',
            [units, this.projection],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.keyBias = this.addWeight(
            `keyBias`,
            [this.projection],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [units, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.valueBias = this.addWeight(
            `valueBias`,
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [units * this.queries, units],
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

            const K = this.applyDense(
                inputs,
                this.keyKernel.read(),
                this.keyBias.read()
            )
            const V = this.applyDense(
                inputs,
                this.valueKernel.read(),
                this.valueBias.read()
            )

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.queries; i++) {
                const Q = this.applyDense(
                    inputs,
                    this.queryKernels[i].read(),
                    this.queryBiases[i].read()
                )

                const scores = tf
                    .matMul(Q, K, false, true)
                    .div(tf.scalar(this.projection).sqrt())
                    .add(mask)

                let weights = scores.softmax()

                weights = kwargs['training']
                    ? tf.dropout(weights, this.dropout)
                    : weights

                const output = tf.matMul(weights, V)
                attentionOutputs.push(output)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            let outputs = this.applyDense(
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
            projection: this.projection,
            queries: this.queries,
            dropout: this.dropout
        }
    }
}

tf.serialization.registerClass(MultiQueryAttention)
