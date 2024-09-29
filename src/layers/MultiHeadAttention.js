import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

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
            this.initializers.glorotUniform()
        )

        // Combined key and value projections for all heads
        this.keyKernel = this.addWeight(
            'keyKernel',
            [units, this.headDim * this.numHeads],
            'float32',
            this.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [units, this.headDim * this.numHeads],
            'float32',
            this.initializers.glorotUniform()
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headDim * this.numHeads * this.queriesPerHead, units],
            'float32',
            this.initializers.glorotUniform()
        )

        if (this.useBias) {
            this.queryBias = this.addWeight(
                'queryBias',
                [this.headDim * this.numHeads * this.queriesPerHead],
                'float32',
                this.initializers.zeros()
            )
            this.keyBias = this.addWeight(
                'keyBias',
                [this.headDim * this.numHeads],
                'float32',
                this.initializers.zeros()
            )
            this.valueBias = this.addWeight(
                'valueBias',
                [this.headDim * this.numHeads],
                'float32',
                this.initializers.zeros()
            )
            this.outputBias = this.addWeight(
                'outputBias',
                [units],
                'float32',
                this.initializers.zeros()
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
                this.numHeads,
                this.queriesPerHead,
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

            // Transpose to [batchSize, numHeads, queriesPerHead, seqLen, headDim]
            const QHeadsTransposed = tf.transpose(QHeads, [0, 2, 3, 1, 4])
            const KHeadsTransposed = tf.transpose(KHeads, [0, 2, 1, 3])
            const VHeadsTransposed = tf.transpose(VHeads, [0, 2, 1, 3])

            // Reshape key and value matrices to be 4D for tiling
            const KHeadsReshaped = tf.reshape(KHeadsTransposed, [
                batchSize,
                this.numHeads,
                seqLen,
                this.headDim
            ])
            const VHeadsReshaped = tf.reshape(VHeadsTransposed, [
                batchSize,
                this.numHeads,
                seqLen,
                this.headDim
            ])

            // Tile key and value matrices to match the number of queries per head
            const KHeadsTiled = tf.tile(KHeadsReshaped, [
                1,
                1,
                this.queriesPerHead,
                1
            ])
            const VHeadsTiled = tf.tile(VHeadsReshaped, [
                1,
                1,
                this.queriesPerHead,
                1
            ])

            // Reshape tiled key and value matrices back to 5D
            const KHeadsTiledReshaped = tf.reshape(KHeadsTiled, [
                batchSize,
                this.numHeads,
                this.queriesPerHead,
                seqLen,
                this.headDim
            ])
            const VHeadsTiledReshaped = tf.reshape(VHeadsTiled, [
                batchSize,
                this.numHeads,
                this.queriesPerHead,
                seqLen,
                this.headDim
            ])

            // Compute attention scores
            let scores = tf.matMul(
                QHeadsTransposed,
                KHeadsTiledReshaped,
                false,
                true
            )
            scores = tf.mul(scores, tf.rsqrt(tf.scalar(this.headDim)))

            // Apply ALiBi if needed
            if (this.ALiBiLength) {
                scores = this.ops.applyALiBi(
                    scores,
                    this.numHeads,
                    this.queriesPerHead,
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
            let outputs = tf.matMul(weights, VHeadsTiledReshaped)

            // Reshape and transpose back
            outputs = tf.transpose(outputs, [0, 3, 1, 2, 4])
            outputs = tf.reshape(outputs, [
                batchSize,
                seqLen,
                this.headDim * this.numHeads * this.queriesPerHead
            ])

            // Final output projection
            outputs = this.ops.applyDense(
                outputs,
                this.outputKernel.read(),
                this.outputBias?.read()
            )

            // Apply dropout if in training mode
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
            numHeads: this.numHeads,
            queriesPerHead: this.queriesPerHead,
            dropout: this.dropout
        }
    }
}

tf.serialization.registerClass(MultiHeadAttention)
