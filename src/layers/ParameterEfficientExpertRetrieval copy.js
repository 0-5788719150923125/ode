import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2407.04153
export default class ParameterEfficientExpertRetrieval extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts || 1000
        this.topK = config.topK || 32
        this.innerDim = config.innerDim || 64
        this.numHeads = config.numHeads || 8
        this.activation = config.activation || 'relu'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        const numSubExperts = Math.floor(Math.sqrt(this.numExperts))

        this.queryKernels = []
        this.subKeys1 = []
        this.subKeys2 = []
        this.expertDownKernels = []
        this.expertUpKernels = []

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel${i}`,
                    [inputDim, this.innerDim],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )

            this.subKeys1.push(
                this.addWeight(
                    `subKeys1_${i}`,
                    [numSubExperts, this.innerDim / 2],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )

            this.subKeys2.push(
                this.addWeight(
                    `subKeys2_${i}`,
                    [numSubExperts, this.innerDim / 2],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )

            this.expertDownKernels.push(
                this.addWeight(
                    `expertDownKernel${i}`,
                    [this.numExperts, inputDim],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )

            this.expertUpKernels.push(
                this.addWeight(
                    `expertUpKernel${i}`,
                    [this.numExperts, inputDim],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const batchSize = inputs.shape[0]
            const inputDim = inputs.shape[1]

            let outputs = tf.zeros([batchSize, inputDim])

            for (let i = 0; i < this.numHeads; i++) {
                const query = tf.matMul(inputs, this.queryKernels[i].read())
                const [subQuery1, subQuery2] = tf.split(query, 2, -1)

                const subTopK1 = this.sparseTopKWithSTE(
                    tf.matMul(subQuery1, this.subKeys1[i].read(), false, true),
                    this.topK / Math.sqrt(this.numExperts)
                )
                const subTopK2 = this.sparseTopKWithSTE(
                    tf.matMul(subQuery2, this.subKeys2[i].read(), false, true),
                    this.topK / Math.sqrt(this.numExperts)
                )

                const expertIndices = this.getExpertIndices(
                    subTopK1.indices,
                    subTopK2.indices
                )
                const expertScores = tf.softmax(
                    tf.add(subTopK1.values, subTopK2.values),
                    -1
                )

                const expertInputs = tf.matMul(
                    inputs,
                    tf.gather(this.expertDownKernels[i].read(), expertIndices)
                )
                const expertOutputs = tf.matMul(
                    tf.relu(expertInputs),
                    tf.gather(this.expertUpKernels[i].read(), expertIndices),
                    false,
                    true
                )

                const weightedOutputs = tf.mul(expertOutputs, expertScores)
                outputs = tf.add(outputs, tf.sum(weightedOutputs, -2))
            }

            return outputs
        })
    }

    sparseTopKWithSTE(inputs, sparsityRatio) {
        return tf.customGrad((inputs, save) => {
            const k = Math.floor(
                inputs.shape[inputs.shape.length - 1] * sparsityRatio
            )
            const topK = tf.topk(tf.abs(inputs), k)
            const mask = tf.greaterEqual(tf.abs(inputs), topK.values.min())
            const sparseOutputs = tf.mul(inputs, tf.cast(mask, inputs.dtype))

            save([inputs, topK.indices])

            return {
                value: sparseOutputs,
                gradFunc: (dy, saved) => {
                    const [inputs, indices] = saved
                    return tf.zerosLike(inputs)
                }
            }
        })(inputs)
    }

    getExpertIndices(indices1, indices2) {
        const batchSize = indices1.shape[0]
        const numSelected = indices1.shape[1]
        const numSubExperts = Math.floor(Math.sqrt(this.numExperts))

        const indX = tf
            .range(0, batchSize)
            .reshape([-1, 1, 1])
            .tile([1, numSelected, 1])
        const indY = indices1.reshape([batchSize, -1, 1])
        const indZ = indices2.reshape([batchSize, 1, -1])

        const indices = tf.reshape(tf.concat([indY, indZ], -1), [
            batchSize,
            numSelected,
            2
        ])
        const expertIndices = tf.add(
            tf.mul(indices.gather([0], 2), numSubExperts),
            indices.gather([1], 2)
        )

        return tf.concat([indX, expertIndices.expandDims(-1)], -1)
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            topK: this.topK,
            innerDim: this.innerDim,
            numHeads: this.numHeads,
            activation: this.activation
        }
    }
}

tf.serialization.registerClass(ParameterEfficientExpertRetrieval)
