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
        this.features = []

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
            this.features.push(
                this.addWeight(
                    `featureMatrix_${i}`,
                    [this.headDim, this.headFeatures],
                    'float32',
                    tf.initializers.orthogonal({ gain: 1.0 }),
                    null,
                    false
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

            const attentionOutputs = []

            for (let i = 0; i < this.numHeads; i++) {
                const Q = this.applyDense(inputs, this.queryKernels[i].read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                const QF = this.applyFeatureMap(Q, this.features[i].read())
                const KF = this.applyFeatureMap(K, this.features[i].read())

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
        const batchSize = KF.shape[0]
        const seqLen = KF.shape[1]
        const dim = KF.shape[2]

        // Create reference tensor
        const ref_v = tf.ones([batchSize, seqLen, 1, dim])

        // Expand dimensions for broadcasting
        const KF_expanded = tf.expandDims(KF, 2)
        const V_expanded = tf.expandDims(V, 2)
        const QF_expanded = tf.expandDims(QF, 1)

        // Compute Gps and Grenorm
        const Gps = tf.mul(KF_expanded, V_expanded)
        const Grenorm = tf.mul(KF_expanded, ref_v)

        // Compute raw attention and normalization
        const att_raw = tf.sum(tf.mul(Gps, QF_expanded), -1)
        const att_norm = tf.sum(tf.mul(Grenorm, QF_expanded), -1)

        // Cumulative sum over the sequence
        const att_raw_cumsum = tf.cumsum(att_raw, 1)
        const att_norm_cumsum = tf.cumsum(att_norm, 1)

        // Normalize
        const att = tf.div(att_raw_cumsum, att_norm_cumsum)

        // Apply attention to values
        const attendedValues = tf.matMul(att, V)

        return attendedValues
    }

    applyFeatureMap(x, randomMatrix) {
        return tf.tidy(() => {
            const projection = tf.matMul(x, randomMatrix)
            const [batchSize, seqLen, featureDim] = projection.shape

            // Reshape to separate features for sin and cos
            const reshaped = tf.reshape(projection, [
                batchSize,
                seqLen,
                featureDim / 2,
                2
            ])

            // Apply sin and cos
            const cosProjection = tf.cos(
                reshaped.slice([0, 0, 0, 0], [-1, -1, -1, 1])
            )
            const sinProjection = tf.sin(
                reshaped.slice([0, 0, 0, 1], [-1, -1, -1, 1])
            )

            // Combine sin and cos
            const combined = tf.reshape(
                tf.stack([cosProjection, sinProjection], 3),
                [batchSize, seqLen, featureDim]
            )

            // Generate random signs (+1 or -1), as used in FAVOR
            const randomSigns = tf
                .randomUniform([1, 1, featureDim], 0, 2, 'int32')
                .mul(2)
                .sub(1)

            // Apply random signs
            const signedCombined = combined.mul(randomSigns)

            // Ensure positivity
            const expCombined = tf.exp(signedCombined)

            // Normalize
            const normalizer = tf.sqrt(tf.scalar(featureDim / 2))
            return expCombined.div(normalizer)
        })
    }

    getWeights() {
        const weights = []

        for (let i = 0; i < this.numHeads; i++) {
            weights.push(this.queryKernels[i].read())
            weights.push(this.keyKernels[i].read())
            weights.push(this.valueKernels[i].read())
            weights.push(this.features[i].read())
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
            this.features[i].write(weights[index++])
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
