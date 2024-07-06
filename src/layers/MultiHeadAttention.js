import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class MultiHeadAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.heads = config.heads || 8
        this.projection = config.projection || 64
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
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.queryBiases.push(
                this.addWeight(
                    `queryBias_${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.keyBiases.push(
                this.addWeight(
                    `keyBias_${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [units, this.projection],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueBiases.push(
                this.addWeight(
                    `valueBias_${i}`,
                    [this.projection],
                    'float32',
                    tf.initializers.zeros(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.projection * this.heads, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputBias = this.addWeight(
            'outputBias',
            [units],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.heads; i++) {
                const Q = this.applyDense(
                    inputs,
                    this.queryKernels[i].read(),
                    this.queryBiases[i].read()
                )
                const K = this.applyDense(
                    inputs,
                    this.keyKernels[i].read(),
                    this.keyBiases[i].read()
                )
                const V = this.applyDense(
                    inputs,
                    this.valueKernels[i].read(),
                    this.valueBiases[i].read()
                )

                const scores = tf
                    .matMul(Q, K, false, true)
                    .div(tf.scalar(Math.sqrt(this.projection)))
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

    getWeights() {
        const weights = []

        for (let i = 0; i < this.heads; i++) {
            weights.push(this.queryKernels[i].read())
            weights.push(this.queryBiases[i].read())
            weights.push(this.keyKernels[i].read())
            weights.push(this.keyBiases[i].read())
            weights.push(this.valueKernels[i].read())
            weights.push(this.valueBiases[i].read())
        }

        weights.push(this.outputKernel.read())
        weights.push(this.outputBias.read())

        return weights
    }

    setWeights(weights) {
        let index = 0

        for (let i = 0; i < this.heads; i++) {
            this.queryKernels[i].write(weights[index++])
            this.queryBiases[i].write(weights[index++])
            this.keyKernels[i].write(weights[index++])
            this.keyBiases[i].write(weights[index++])
            this.valueKernels[i].write(weights[index++])
            this.valueBiases[i].write(weights[index++])
        }

        this.outputKernel.write(weights[index++])
        this.outputBias.write(weights[index])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            heads: this.heads,
            projection: this.projection,
            dropout: this.dropout
        }
    }
}

tf.serialization.registerClass(MultiHeadAttention)
