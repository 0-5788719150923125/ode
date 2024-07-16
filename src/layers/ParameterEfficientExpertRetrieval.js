import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ParameterEfficientExpertRetrieval extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts || 1048576 // 1024^2
        this.expertDim = config.expertDim || 1 // Single neuron experts
        this.numHeads = config.numHeads || 8
        this.topK = config.topK || 16
        this.innerDim = config.innerDim || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.numSubExperts = Math.ceil(Math.sqrt(this.numExperts))

        this.queryNetworks = []
        this.subKeys1 = []
        this.subKeys2 = []
        this.expertWeights = []
        this.projectionMatrices = []

        for (let i = 0; i < this.numHeads; i++) {
            this.queryNetworks.push(
                this.addWeight(
                    `query_${i}`,
                    [inputDim, this.innerDim],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )
            this.subKeys1.push(
                this.addWeight(
                    `subKeys1_${i}`,
                    [this.numSubExperts, this.innerDim / 2],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )
            this.subKeys2.push(
                this.addWeight(
                    `subKeys2_${i}`,
                    [this.numSubExperts, this.innerDim / 2],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )
            this.expertWeights.push(
                this.addWeight(
                    `expertWeights_${i}`,
                    [this.numExperts, inputDim],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )
            this.projectionMatrices.push(
                this.addWeight(
                    `projection_${i}`,
                    [this.topK, inputDim],
                    'float32',
                    tf.initializers.glorotNormal()
                )
            )
        }

        this.batchNorm = tf.layers.batchNormalization({ axis: -1 })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const batchSize = inputs.shape[0]
            const inputDim = inputs.shape[1]
            let outputs = tf.zeros([batchSize, inputDim, inputDim])

            for (let i = 0; i < this.numHeads; i++) {
                // Query projection and batch normalization
                let query = this.ops.applyDense(
                    inputs,
                    this.queryNetworks[i].read()
                )
                query = this.batchNorm.apply(query)

                // Split query for product key retrieval
                const [query1, query2] = tf.split(query, 2, -1)

                // Compute similarities with sub-keys
                const similarities1 = this.ops.applyDense(
                    query1,
                    this.subKeys1[i].read().transpose()
                )
                const similarities2 = this.ops.applyDense(
                    query2,
                    this.subKeys2[i].read().transpose()
                )

                // Get top-k indices and scores
                const [topIndices1, topScores1] = this.getTopK(
                    similarities1,
                    this.topK
                )
                const [topIndices2, topScores2] = this.getTopK(
                    similarities2,
                    this.topK
                )

                // Combine indices to get expert indices
                const expertIndices = this.getExpertIndices(
                    topIndices1,
                    topIndices2
                )

                // Combine scores
                const scores = tf.add(topScores1, topScores2)
                const normalizedScores = tf.softmax(scores, -1)

                // Gather expert weights and apply
                const selectedExperts = tf.gather(
                    this.expertWeights[i].read(),
                    expertIndices
                )

                // Apply expert weights to inputs
                const expertOutputs = tf.matMul(
                    inputs.expandDims(1),
                    selectedExperts.transpose([0, 1, 3, 2])
                )

                // Weight outputs by scores and sum
                const weightedOutputs = tf.mul(
                    expertOutputs,
                    normalizedScores.expandDims(-2)
                )
                const summedOutputs = tf.sum(weightedOutputs, 1)

                // Project back to original dimension
                const projectedOutputs = this.ops.applyDense(
                    summedOutputs,
                    this.projectionMatrices[i].read()
                )
                outputs = tf.add(outputs, projectedOutputs)
                console.log(Math.random())
            }

            // Add residual connection
            outputs = tf.add(outputs, inputs.expandDims(-1))

            return outputs
        })
    }

    getTopK(x, k) {
        const values = tf.topk(x, k).values
        const indices = tf.topk(x, k).indices
        return [indices, values]
    }

    getExpertIndices(indices1, indices2) {
        return tf.tidy(() => {
            const combinedIndices = tf.add(
                tf.mul(indices1, this.numSubExperts),
                indices2
            )
            // Ensure indices are within the valid range
            return tf.cast(
                tf.minimum(
                    combinedIndices,
                    tf.scalar(this.numExperts - 1, 'int32')
                ),
                'int32'
            )
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            expertDim: this.expertDim,
            numHeads: this.numHeads,
            topK: this.topK,
            innerDim: this.innerDim
        }
    }
}

tf.serialization.registerClass(ParameterEfficientExpertRetrieval)
