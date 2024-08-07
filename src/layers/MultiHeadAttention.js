import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class MultiHeadAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 256
        this.queriesPerHead = config.queriesPerHead || 1
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        // Combined query projections for all heads
        this.queryKernel = this.addWeight(
            'queryKernel',
            [units, this.headDim * this.numHeads * this.queriesPerHead],
            'float32',
            tf.initializers.glorotUniform()
        )

        // Combined key and value projections for all heads
        this.keyKernel = this.addWeight(
            'keyKernel',
            [units, this.headDim * this.numHeads],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [units, this.headDim * this.numHeads],
            'float32',
            tf.initializers.glorotUniform()
        )

        if (this.useBias) {
            this.queryBias = this.addWeight(
                'queryBias',
                [this.headDim * this.numHeads * this.queriesPerHead],
                'float32',
                tf.initializers.zeros()
            )
            this.keyBias = this.addWeight(
                'keyBias',
                [this.headDim * this.numHeads],
                'float32',
                tf.initializers.zeros()
            )
            this.valueBias = this.addWeight(
                'valueBias',
                [this.headDim * this.numHeads],
                'float32',
                tf.initializers.zeros()
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headDim * this.numHeads * this.queriesPerHead, units],
            'float32',
            tf.initializers.glorotUniform()
        )
        if (this.useBias) {
            this.outputBias = this.addWeight(
                'outputBias',
                [units],
                'float32',
                tf.initializers.zeros()
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLen, inputDim] = inputs.shape

            // Compute Q, K, V for all heads in parallel
            const Q = this.ops.applyDense(
                inputs,
                this.queryKernel.read(),
                this.queryBias?.read()
            )
            const K = this.ops.applyDense(
                inputs,
                this.keyKernel.read(),
                this.keyBias?.read()
            )
            const V = this.ops.applyDense(
                inputs,
                this.valueKernel.read(),
                this.valueBias?.read()
            )

            // Reshape Q, K, V to split into heads
            const QHeads = tf.reshape(Q, [
                batchSize,
                seqLen,
                this.numHeads * this.queriesPerHead,
                this.headDim
            ])
            const KHeads = tf.reshape(K, [
                batchSize,
                seqLen,
                this.numHeads,
                this.headDim
            ])
            const VHeads = tf.reshape(V, [
                batchSize,
                seqLen,
                this.numHeads,
                this.headDim
            ])

            // Transpose to [batchSize, numHeads, seqLen, headDim]
            const QHeadsTransposed = tf.transpose(QHeads, [0, 2, 1, 3])
            const KHeadsTransposed = tf.transpose(KHeads, [0, 2, 1, 3])
            const VHeadsTransposed = tf.transpose(VHeads, [0, 2, 1, 3])

            // Compute attention scores
            let scores = tf.matMul(
                QHeadsTransposed,
                KHeadsTransposed,
                false,
                true
            )
            scores = tf.div(scores, tf.sqrt(tf.scalar(this.headDim)))

            // Apply ALiBi if needed
            if (this.ALiBiLength) {
                scores = this.ops.applyALiBi(
                    scores,
                    this.numHeads,
                    this.ALiBiLength
                )
            }

            // Apply causal mask
            const mask = tf.linalg
                .bandPart(tf.ones([seqLen, seqLen]), 0, -1)
                .sub(tf.eye(seqLen))
                .mul(tf.scalar(-1e9))
            scores = tf.add(scores, mask)

            // Compute attention weights
            let weights = tf.softmax(scores)
            weights = kwargs['training']
                ? tf.dropout(weights, this.dropout)
                : weights

            // Apply attention weights to values
            let output = tf.matMul(weights, VHeadsTransposed)

            // Reshape and transpose back
            output = tf.transpose(output, [0, 2, 1, 3])
            output = tf.reshape(output, [
                batchSize,
                seqLen,
                this.headDim * this.numHeads * this.queriesPerHead
            ])

            // Final output projection
            output = this.ops.applyDense(
                output,
                this.outputKernel.read(),
                this.outputBias?.read()
            )

            // Apply normalization and residual connection
            output = this.ops.rmsNorm(output)
            output = tf.add(inputs, output)

            // Apply dropout if in training mode
            output = kwargs['training']
                ? tf.dropout(output, this.dropout)
                : output

            return output
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            headDim: this.headDim,
            numHeads: this.numHeads,
            queriesPerHead: this.queriesPerHead,
            dropout: this.dropout
        }
    }
}

tf.serialization.registerClass(MultiHeadAttention)
