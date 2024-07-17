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
        this.maxPenalty = 0.1
        this.epsilon = 1e-6
        this.expertUsageCounts = tf.variable(
            tf.zeros([this.topK, this.numExperts]),
            false
        )
        this.totalUsage = tf.variable(tf.scalar(0), false)
        this.debug = true
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.switchingHidden = this.addWeight(
            'switchingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotNormal()
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
            tf.initializers.glorotNormal()
        )
        this.switchingBias = this.addWeight(
            'switchingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros()
        )
        this.outputProjection = this.addWeight(
            'outputProjection',
            [this.topK * inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const outputs = this.approximateTopKWithGumbel(
                inputs,
                this.topK,
                kwargs
            )
            return this.ops.rmsNorm(outputs)
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

    computeUtilization(expertIndices, kwargs) {
        const usageCounts = this.expertUsageCounts
        const totalUsage = this.totalUsage

        const expertUtilization = usageCounts.div(totalUsage.add(this.epsilon))

        const targetUtilization = tf.fill(
            [this.numExperts],
            1 / this.numExperts
        )

        const utilizationDiff = expertUtilization.sub(targetUtilization).abs()
        const expertDiversityLoss = utilizationDiff.mean()

        const avgUsage = totalUsage.div(this.numExperts)
        const usageDeviations = usageCounts
            .sub(avgUsage)
            .abs()
            .div(avgUsage.add(this.epsilon))
        const loadBalancingLoss = usageDeviations.mean()

        const combinedLoss = expertDiversityLoss.add(loadBalancingLoss).div(2)

        const scaledLoss = combinedLoss.mul(this.maxPenalty)

        if (this.debug) {
            console.log('Start of computeUtilization')
            console.log('usageCounts:', usageCounts.arraySync())
            console.log('totalUsage:', totalUsage.arraySync())
            console.log('expertUtilization:', expertUtilization.arraySync())
            console.log('targetUtilization:', targetUtilization.arraySync())
            console.log('expertDiversityLoss:', expertDiversityLoss.arraySync())
            console.log('loadBalancingLoss:', loadBalancingLoss.arraySync())
            console.log('combinedLoss:', combinedLoss.arraySync())
            console.log('scaledLoss:', scaledLoss.arraySync())
        }

        this.extraLoss = tf.keep(scaledLoss)

        this.updateExpertUsage(expertIndices)

        return scaledLoss
    }

    approximateTopKWithGumbel(inputs, k, kwargs) {
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
        const expertWeights = this.ops.gumbelSoftmax(
            switchingScores,
            this.temperature
        )

        let expertIndices
        const expertValues = tf.customGrad((expertWeights, save) => {
            const topKWeights = tf.topk(expertWeights, k)
            const topKIndices = tf.topk(expertWeights.sum(1), k)
            expertIndices = topKIndices.indices.arraySync()
            if (kwargs.training) this.computeUtilization(topKIndices.indices)
            save([expertWeights, topKIndices.indices])
            return {
                value: topKWeights.values,
                gradFunc: (dy, saved) => {
                    // const [expertWeights] = saved
                    // const tileShape = [1, 1, expertWeights.shape[2] / k]
                    // return [dy.tile(tileShape).add(expertWeights)]
                    const [expertWeights, topKIndices] = saved
                    const gradientMask = tf.zeros(expertWeights.shape)

                    for (let i = 0; i < inputs.shape[0]; i++) {
                        for (let j = 0; j < k; j++) {
                            const batchIndex = i
                            // console.log(topKIndices.indices)
                            const expertIndex = topKIndices
                                .slice([i, j], [1, 1])
                                .flatten()
                                .arraySync()[0]
                            console.log(expertIndex)
                            const updateMask = tf.zeros(expertWeights.shape)
                            const updateMaskBuffer = updateMask.bufferSync()
                            console.log('update mask:', updateMaskBuffer)
                            const updateMaskSlice = updateMask.slice(
                                [batchIndex, 0, expertIndex],
                                [1, -1, 1]
                            )
                            const updateMaskSliceBuffer = updateMaskBuffer.set(
                                dy.slice([i, 0, j], [1, -1, 1]),
                                updateMaskSlice
                            )
                            gradientMask.assign(gradientMask.add(updateMask))
                        }
                    }

                    return [gradientMask]
                }
            }
        })(expertWeights)

        const batchOutputs = []
        for (let i = 0; i < inputs.shape[0]; i++) {
            const batchInputs = inputs.slice([i, 0], [1, -1])

            const expertOutputs = []
            for (let j = 0; j < k; j++) {
                const expertIndex = expertIndices[i][j]
                const expertValue = expertValues.slice([i, 0, j], [1, -1, 1])
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
