import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class AdaptiveMixtureOfExperts extends LayerBase {
    constructor(config) {
        super(config)
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.topK = config.topK || 2
        this.switchingDim = config?.switchingDim || 64
        this.activation = config.activation || 'swish'
        this.temperature = config.temperature || 1.0
        this.step = 0
        this.maxPenalty = config.maxPenalty || 0.1
        this.rampUpSteps = config.rampUpSteps || 100
        this.epsilon = 1e-7
        this.expertUsageCounts = tf.variable(
            tf.zeros([this.topK, this.numExperts]),
            false
        )
        this.totalUsage = tf.variable(tf.scalar(0), false)
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.switchingHidden = this.addWeight(
            'switchingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingHiddenBias = this.addWeight(
            'switchingHiddenBias',
            [this.switchingDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingKernel = this.addWeight(
            'switchingKernel',
            [this.switchingDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingBias = this.addWeight(
            'switchingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputProjection = this.addWeight(
            'outputProjection',
            [this.topK * inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            return this.approximateTopKWithGumbel(inputs, this.topK)
        })
    }

    updateExpertUsage(selectedExperts) {
        const batchUsage = tf.sum(
            tf.oneHot(selectedExperts, this.numExperts),
            0
        )
        this.expertUsageCounts.assign(this.expertUsageCounts.add(batchUsage))
        this.totalUsage.assign(this.totalUsage.add(tf.sum(batchUsage)))
    }

    computeUtilization(expertIndices) {
        return tf.tidy(() => {
            console.log('Start of computeUtilization')

            const usageCounts = this.expertUsageCounts
            const totalUsage = this.totalUsage

            console.log('usageCounts:', usageCounts.arraySync())
            console.log('totalUsage:', totalUsage.arraySync())

            const expertUtilization = usageCounts.div(
                totalUsage.add(this.epsilon)
            )
            console.log('expertUtilization:', expertUtilization.arraySync())

            const targetUtilization = tf.fill(
                [this.numExperts],
                1 / this.numExperts
            )
            console.log('targetUtilization:', targetUtilization.arraySync())

            const utilizationDiff = expertUtilization
                .sub(targetUtilization)
                .abs()
            const expertDiversityLoss = utilizationDiff.mean()
            console.log('expertDiversityLoss:', expertDiversityLoss.arraySync())

            const avgUsage = totalUsage.div(this.numExperts)
            const usageDeviations = usageCounts
                .sub(avgUsage)
                .abs()
                .div(avgUsage.add(this.epsilon))
            const loadBalancingLoss = usageDeviations.mean()
            console.log('loadBalancingLoss:', loadBalancingLoss.arraySync())

            const combinedLoss = expertDiversityLoss
                .add(loadBalancingLoss)
                .div(2)
            console.log('combinedLoss:', combinedLoss.arraySync())

            this.step++
            const rampUpFactor = Math.min(this.step / this.rampUpSteps, 1)
            console.log('step:', this.step, 'rampUpFactor:', rampUpFactor)

            const scaledLoss = combinedLoss.mul(rampUpFactor * this.maxPenalty)
            console.log('scaledLoss:', scaledLoss.arraySync())

            this.extraLoss = tf.keep(scaledLoss)

            return scaledLoss
        })
    }

    approximateTopKWithGumbel(inputs, k) {
        const switchingHidden = this.ops.applyDense(
            inputs,
            this.switchingHidden.read(),
            this.switchingHiddenBias.read()
        )
        const switchingActivated = tf.layers
            .activation({ activation: this.activation })
            .apply(switchingHidden)
        const switchingScores = this.ops.applyDense(
            switchingActivated,
            this.switchingKernel.read(),
            this.switchingBias.read()
        )
        // console.log(switchingScores)
        const expertWeights = this.ops.gumbelSoftmax(
            switchingScores.sum(1),
            this.temperature
        )

        let expertIndices
        const expertValues = tf.customGrad((expertWeights, save) => {
            const topKWeights = tf.topk(expertWeights, k)
            this.updateExpertUsage(topKWeights.indices)
            this.computeUtilization(expertIndices)
            expertIndices = topKWeights.indices.arraySync()
            save([expertWeights])
            return {
                value: topKWeights.values,
                gradFunc: (dy, saved) => {
                    const [expertWeights] = saved
                    const tileShape = [1, expertWeights.shape[1] / k]
                    return [dy.tile(tileShape)]
                }
            }
        })(expertWeights)

        const batchOutputs = []
        for (let i = 0; i < inputs.shape[0]; i++) {
            const batchInputs = inputs.slice([i, 0], [1, -1])
            const batchExpertIndices = expertIndices[i]
            const batchExpertValues = expertValues.slice([i, 0], [1, -1])

            const expertOutputs = []
            for (let j = 0; j < k; j++) {
                const expertIndex = batchExpertIndices[j]
                const expertValue = batchExpertValues.slice([0, j], [1, 1])
                const expertOutput =
                    this.experts[expertIndex].apply(batchInputs)
                expertOutputs.push(expertOutput.mul(expertValue))
            }

            batchOutputs.push(tf.concat(expertOutputs, -1))
        }

        const outputProjected = this.ops.applyDense(
            tf.concat(batchOutputs, 0),
            this.outputProjection.read()
        )

        let outputWeighted = outputProjected
        for (const i in expertIndices) {
            for (const j of expertIndices[i]) {
                const expertScore = switchingScores.slice([i, 0, j], [1, -1, 1])
                outputWeighted = outputWeighted.mul(expertScore)
            }
        }

        return outputProjected
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            switchingDim: this.switchingDim,
            activation: this.activation,
            topK: this.topK,
            temperature: this.temperature
        }
    }
}

tf.serialization.registerClass(AdaptiveMixtureOfExperts)
