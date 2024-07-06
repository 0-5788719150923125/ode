import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// Loosely-inspired by Performer:
// https://arxiv.org/abs/2009.14794
export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 256
        this.numFeatures = config.numFeatures || 256
        this.numHeads = config.numHeads || 8
        this.headDim = Math.floor(this.hiddenDim / this.numHeads)
        this.useALiBi = config.useALiBi || false
        this.epsilon = 1e-6
        if (this.hiddenDim % this.numHeads !== 0) {
            throw new Error(
                `hiddenDim (${this.headDim}) should be divisible by numHeads (${this.numHeads})!`
            )
        }
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.inputDim = inputDim

        // Create weight matrices for each head
        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []
        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [inputDim, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
        }

        this.outputKernel = this.addWeight(
            `outputKernel`,
            [this.hiddenDim, inputDim],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        this.randomMatrix = tf.randomNormal(
            [this.headDim, this.numFeatures],
            0,
            1 / Math.sqrt(this.numFeatures)
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            // Ensure inputs is a tensor, not an array
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLen, features] = inputs.shape

            // Process each head
            const headOutputs = this.queryKernels.map((queryKernel, i) => {
                // Linear transformations to create query, key, and value for this head
                const Q = this.applyDense(inputs, queryKernel.read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                // Apply random feature map to query and key
                const QF = this.applyFeatureMap(Q)
                const KF = this.applyFeatureMap(K)

                // Compute key-value representation
                const KFV = tf.matMul(KF, V, true, false)
                // Compute normalization factor
                const D = tf.sum(KF, -2, true)

                // Compute attention scores
                let QF_KFV = tf.matMul(QF, KFV)

                if (this.useALiBi) {
                    QF_KFV = this.applyALiBi(QF_KFV, this.numHeads, i, seqLen)
                }

                const mask = tf.tidy(() => {
                    const headSeqLen = QF_KFV.shape[1]
                    const headFeatures = QF_KFV.shape[2]

                    const baseMask = tf.linalg.bandPart(
                        tf.ones([headSeqLen, headFeatures]),
                        0,
                        -1
                    )

                    const identityMask = tf.eye(
                        Math.min(headSeqLen, headFeatures)
                    )

                    const paddedIdentityMask = identityMask.pad([
                        [0, Math.max(0, headSeqLen - headFeatures)],
                        [0, Math.max(0, headFeatures - headSeqLen)]
                    ])

                    const combinedMask = baseMask
                        .sub(paddedIdentityMask)
                        .mul(tf.scalar(-1e9))

                    const expandedMask = combinedMask.expandDims(0)

                    const tiledMask = expandedMask.tile([batchSize, 1, 1])

                    return tiledMask
                })

                const maskedScores = QF_KFV.add(mask)

                // Compute normalization term via element-wise multiplication for efficient broadcasting
                const QF_D = tf.mul(QF, D)
                // Sum over the feature dimension
                const QF_D_sum = tf.sum(QF_D, -1, true)

                // Implementation of attention mechanism with epsilon for numerical stability
                return tf.div(maskedScores, tf.add(QF_D_sum, this.epsilon))
            })

            // Concatenate head outputs
            let outputs = tf.concat(headOutputs, -1)

            // Apply layer normalization
            outputs = this.ops.rmsNorm(outputs)

            // Apply output projection
            outputs = this.applyDense(outputs, this.outputKernel.read())
            // Scale down outputs for stability
            outputs = tf.mul(outputs, tf.scalar(0.1))

            // Apply residual connection
            return tf.add(inputs, outputs)
        })
    }

    applyFeatureMap(x) {
        const projection = tf.matMul(x, this.randomMatrix)
        // ReLU activation for sparsity and efficiency
        return tf.relu(projection)
    }

    getWeights() {
        return [
            ...this.queryKernels.map((k) => k.read()),
            ...this.keyKernels.map((k) => k.read()),
            ...this.valueKernels.map((k) => k.read()),
            this.outputKernel.read(),
            this.randomMatrix
        ]
    }

    setWeights(weights) {
        const headWeights = weights.slice(0, -1)
        const numHeadWeights = headWeights.length
        const weightsPerHead = numHeadWeights / 3

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels[i].write(headWeights[i])
            this.keyKernels[i].write(headWeights[i + weightsPerHead])
            this.valueKernels[i].write(headWeights[i + 2 * weightsPerHead])
        }

        this.outputKernel.write(weights[weights.length - 2])
        this.randomMatrix.write(weights[weights.length - 1])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            numFeatures: this.numFeatures,
            numHeads: this.numHeads,
            useALiBi: this.useALiBi
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureAttention)
