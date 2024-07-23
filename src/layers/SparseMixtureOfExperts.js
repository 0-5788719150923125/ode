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

            const { expertIndices, expertWeights } = this.topKWithGumbel(
                switchingScores,
                this.topK
            )

            if (kwargs.training) this.computeUtilization(expertIndices)

            const allOutputs = []
            const rawIndices = expertIndices.arraySync()
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

    topKWithGumbel(scores, k) {
        const gumbel = this.ops.gumbelSoftmax(scores.mean(1), this.temperature)

        const numExperts = gumbel.shape[1]

        const expertIndices = tf.customGrad((gumbel, save) => {
            const { indices, values } = tf.topk(gumbel, k)
            save([gumbel, indices])
            return {
                value: indices,
                gradFunc: (dy, [gumbel, indices]) => {
                    // Create a gradient of the same shape as gumbel
                    const fullGradient = tf.buffer(gumbel.shape)

                    // Scatter the gradient from dy into the full gradient
                    indices.bufferSync().values.forEach((index, i) => {
                        const batchIdx = Math.floor(i / k)
                        const gradValue = dy.bufferSync().get(batchIdx, i % k)
                        fullGradient.set(gradValue, batchIdx, index)
                    })

                    // Convert buffer to tensor and multiply element-wise with gumbel
                    return [fullGradient.toTensor().mul(gumbel)]
                }
            }
        })(gumbel)

        const oneHotIndices = tf.oneHot(expertIndices, numExperts)

        const expertWeights = this.ops
            .applyDense(
                oneHotIndices,
                this.expertWeights.read(),
                this.expertBiases.read()
            )
            .softmax()

        return { expertIndices, expertWeights }
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
