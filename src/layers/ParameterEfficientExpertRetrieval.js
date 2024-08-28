import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2407.04153
export default class ParameterEfficientExpertRetrieval extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts || 1024
        this.numHeads = config.numHeads || 8
        this.topK = config.topK || 32
        this.queryDim = config.queryDim || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        const outputDim = inputDim

        this.queryWeights = []
        for (let i = 0; i < this.numHeads; i++) {
            this.queryWeights.push(
                this.addWeight(
                    `queryWeight${i}`,
                    [this.queryDim, inputDim],
                    'float32',
                    tf.initializers.glorotNormal({
                        seed: this.ops.getSeed()
                    }),
                    null,
                    true
                )
            )
        }

        this.expertWeights = this.addWeight(
            `expertWeights`,
            [this.numExperts, inputDim + outputDim],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            }),
            null,
            true
        )
    }

    sparseTopKWithSTE(inputs) {
        return tf.customGrad((inputs, save) => {
            const { indices, values } = tf.topk(inputs, this.topK)
            save([indices])

            return {
                value: indices,
                gradFunc: (dy, saved) => {
                    const [indices] = saved
                    const mask = tf.oneHot(indices.flatten(), inputs.shape[1])
                    return [tf.mul(dy, mask)]
                }
            }
        })(inputs)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const batchSize = inputs.shape[0]
            const inputDim = inputs.shape[inputs.shape.length - 1]

            console.log(`Input shape: ${inputs.shape}`)

            const queries = this.queryWeights.map((queryWeight) =>
                tf.matMul(inputs, queryWeight.read(), false, true)
            )

            console.log(`Query shape: ${queries[0].shape}`)

            const expertWeights = this.expertWeights.read()
            console.log(`Expert weights shape: ${expertWeights.shape}`)

            const scores = queries.map((query) => {
                const expertScores = tf.matMul(
                    query,
                    expertWeights
                        .slice([0, 0], [this.numExperts, inputDim])
                        .expandDims(0),
                    false,
                    true
                )
                console.log(`Expert scores shape: ${expertScores.shape}`)
                return expertScores
            })

            const expertIndices = scores.map((score) =>
                this.sparseTopKWithSTE(score)
            )

            console.log(`Expert indices shape: ${expertIndices[0].shape}`)

            const selectedExpertWeights = expertIndices.map((indices) =>
                tf
                    .gather(expertWeights, indices)
                    .reshape([batchSize, this.topK, -1])
            )

            console.log(
                `Selected expert weights shape: ${selectedExpertWeights[0].shape}`
            )

            const expertOutputs = selectedExpertWeights.map((weights) => {
                const weightsDown = weights.slice(
                    [0, 0, 0],
                    [batchSize, this.topK, inputDim]
                )
                const weightsUp = weights.slice(
                    [0, 0, inputDim],
                    [batchSize, this.topK, -1]
                )

                console.log(`Weights down shape: ${weightsDown.shape}`)
                console.log(`Weights up shape: ${weightsUp.shape}`)

                const intermediate = tf.matMul(inputs, weightsDown, false, true)
                console.log(`Intermediate shape: ${intermediate.shape}`)

                return tf.matMul(tf.relu(intermediate), weightsUp)
            })

            return expertOutputs.reduce((a, b) => tf.add(a, b))
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            numHeads: this.numHeads,
            topK: this.topK,
            queryDim: this.queryDim
        }
    }
}

tf.serialization.registerClass(ParameterEfficientExpertRetrieval)
