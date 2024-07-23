import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SparseMixtureOfExperts extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts
        this.topK = config.topK || 2
        this.switchingDim = config.switchingDim || 128
        this.activation = config.activation || 'swish'
        this.temperature = 0.8
        this.epsilon = 1e-8
        this.expertUsage = tf.variable(tf.zeros([this.numExperts]), false)
        this.totalUsage = tf.variable(tf.scalar(0), false)
        this.debug = false
    }

    build(inputShape) {
        const inputDim = inputShape[0][inputShape[0].length - 1]

        this.switchingHidden = this.addWeight(
            'switchingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.switchingHiddenBias = this.addWeight(
            'switchingHiddenBias',
            [this.switchingDim],
            'float32',
            tf.initializers.zeros()
        )
        this.switchingKernel = this.addWeight(
            'switchingKernel',
            [this.switchingDim, this.numExperts],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.switchingBias = this.addWeight(
            'switchingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros()
        )
        this.expertWeights = this.addWeight(
            'expertWeights',
            [this.numExperts, inputDim],
            'float32',
            tf.initializers.randomNormal({
                mean: 1.0,
                stddev: 0.1
            })
        )
        this.expertBiases = this.addWeight(
            'expertBiases',
            [this.topK, inputDim],
            'float32',
            tf.initializers.zeros()
        )
        this.outputProjection = this.addWeight(
            'outputProjection',
            [this.topK * inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const expertOutputs = inputs.slice(1)
            inputs = inputs[0]

            const switchingHidden = this.ops.applyDense(
                inputs,
                this.switchingHidden.read(),
                this.switchingHiddenBias.read()
            )
            const switchingGate = tf.layers
                .activation({ activation: this.activation })
                .apply(switchingHidden)
            const switchingScores = this.ops.applyDense(
                switchingGate,
                this.switchingKernel.read(),
                this.switchingBias.read()
            )

            const { discreteIndices, expertWeights } = this.topKWithGumbel(
                switchingScores,
                this.topK,
                this.numExperts
            )

            if (kwargs.training) this.computeUtilization(discreteIndices)

            const allOutputs = []
            const rawIndices = discreteIndices.arraySync()

            for (let i = 0; i < rawIndices.length; i++) {
                const batchOutputs = []
                for (let j = 0; j < this.topK; j++) {
                    const expertIndex = rawIndices[i][j]
                    const expertOutput = expertOutputs[expertIndex].slice(
                        [i, 0, 0],
                        [1, -1, -1]
                    )
                    const expertWeight = expertWeights.slice(
                        [i, j, 0],
                        [1, 1, -1]
                    )
                    batchOutputs.push(expertOutput.mul(expertWeight))
                }
                allOutputs.push(tf.concat(batchOutputs, -1))
            }

            const outputProjected = this.ops.applyDense(
                tf.concat(allOutputs, 0),
                this.outputProjection.read()
            )

            return outputProjected
        })
    }

    topKWithGumbel(scores, k, numExperts) {
        const expertIndices = tf.customGrad((scores, save) => {
            // Forward pass: Use hard top-k
            const meanScores = scores.mean(1)
            const { indices, values } = tf.topk(meanScores, k)

            // Create one-hot representation for the forward pass
            const oneHotIndices = tf.oneHot(indices, numExperts)

            save([scores, indices, oneHotIndices])

            return {
                value: oneHotIndices,
                gradFunc: (dy, [scores, indices, oneHotIndices]) => {
                    // Backward pass: Use Gumbel-Softmax
                    const gumbelProbs = this.ops.gumbelSoftmax(
                        scores,
                        this.temperature
                    )

                    // Create a mask based on the top-k indices
                    const mask = oneHotIndices.expandDims(1)

                    // Reshape dy to align with gumbelProbs
                    const dyReshaped = dy
                        .expandDims(1)
                        .tile([1, scores.shape[1], 1, 1])

                    // Compute gradients
                    const maskedProbs = gumbelProbs.expandDims(2).mul(mask)

                    // Integrate the derivative with Gumbel approximation
                    const grads = maskedProbs.mul(dyReshaped)

                    // Sum over the top-k dimension
                    const summedGrads = grads.sum(2)

                    // Normalize the gradients
                    const normalizedGrads = summedGrads.div(
                        summedGrads.sum(-1, true).add(this.epsilon)
                    )

                    return [normalizedGrads.sigmoid()]
                }
            }
        })(scores)

        const expertWeights = this.ops
            .applyDense(
                expertIndices,
                this.expertWeights.read(),
                this.expertBiases.read()
            )
            .softmax()

        const discreteIndices = tf.argMax(expertIndices, -1)

        return { discreteIndices, expertWeights }
    }

    computeUtilization(expertIndices, kwargs) {
        const expertUtilization = this.expertUsage.div(
            this.totalUsage.add(this.epsilon)
        )
        const targetUtilization = tf.fill(
            [this.numExperts],
            1 / this.numExperts
        )

        const utilizationDiff = expertUtilization.sub(targetUtilization).abs()
        const expertDiversityLoss = utilizationDiff.mean()

        const avgUsage = this.totalUsage.div(this.numExperts)
        const usageDeviations = this.expertUsage
            .sub(avgUsage)
            .abs()
            .div(avgUsage.add(this.epsilon))

        const loadBalancingLoss = usageDeviations.mean()

        const combinedLoss = expertDiversityLoss.add(loadBalancingLoss).div(2)

        if (this.debug) {
            console.log('Start of computeUtilization')
            console.log('expertUsage:', this.expertUsage.arraySync())
            console.log('totalUsage:', this.totalUsage.arraySync())
            console.log('expertUtilization:', expertUtilization.arraySync())
            console.log('targetUtilization:', targetUtilization.arraySync())
            console.log('expertDiversityLoss:', expertDiversityLoss.arraySync())
            console.log('loadBalancingLoss:', loadBalancingLoss.arraySync())
            console.log('combinedLoss:', combinedLoss.arraySync())
        }

        this.extraLoss = tf.keep(combinedLoss)

        this.updateExpertUsage(expertIndices.flatten())

        return combinedLoss
    }

    updateExpertUsage(selectedExperts) {
        const batchUsage = tf.sum(
            tf.oneHot(selectedExperts, this.numExperts),
            0
        )
        this.expertUsage.assign(this.expertUsage.add(batchUsage))
        this.totalUsage.assign(this.totalUsage.add(tf.sum(batchUsage)))
    }

    computeOutputShape(inputShape) {
        return inputShape[0]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            switchingDim: this.switchingDim,
            activation: this.activation,
            topK: this.topK
        }
    }
}

tf.serialization.registerClass(SparseMixtureOfExperts)
