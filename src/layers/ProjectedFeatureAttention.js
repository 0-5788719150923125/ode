import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 256
        this.headFeatures = config.headFeatures || 64
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
            this.features.push(
                this.addWeight(
                    `featureMatrix_${i}`,
                    [this.headDim, this.headFeatures],
                    'float32',
                    tf.initializers.randomNormal({
                        mean: 0,
                        stddev: 1 / Math.sqrt(this.headFeatures)
                    }),
                    null,
                    false
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headDim * this.numHeads, inputDim],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.numHeads; i++) {
                const Q = this.applyDense(inputs, this.queryKernels[i].read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                // Apply random feature map to query and key
                const QF = this.applyFeatureMap(Q, this.features[i].read())
                const KF = this.applyFeatureMap(K, this.features[i].read())

                const scores = tf
                    .matMul(QF, KF, false, true)
                    .div(tf.scalar(Math.sqrt(this.headFeatures)))
                    .add(mask)

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

    applyFeatureMap(x, featureMatrix) {
        const projection = tf.matMul(x, featureMatrix)
        // ReLU activation for sparsity and efficiency
        return tf.relu(projection)
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
