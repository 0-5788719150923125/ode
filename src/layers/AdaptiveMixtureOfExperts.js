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
        this.epsilon = 1e-7
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
        // this.projectionMatrix = this.addWeight(
        //     'projectionMatrix',
        //     [inputDim, this.numExperts],
        //     'float32',
        //     tf.initializers.randomNormal({
        //         mean: 0,
        //         stdDev: 1.0,
        //         seed: 42
        //     }),
        //     null,
        //     false
        // )
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
        this.step++

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
            console.log('step:', this.step)
            console.log('scaledLoss:', scaledLoss.arraySync())
        }

        this.extraLoss = tf.keep(scaledLoss)

        this.updateExpertUsage(expertIndices)

        return scaledLoss
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
        const expertWeights = this.ops.gumbelSoftmax(
            switchingScores,
            this.temperature
        )
        // .add(this.epsilon)

        let expertIndices
        const expertValues = tf.customGrad((expertWeights, save) => {
            const topKWeights = tf.topk(expertWeights, k)
            const topKIndices = tf.topk(expertWeights.sum(1), k)
            expertIndices = topKIndices.indices.arraySync()
            this.computeUtilization(topKIndices.indices)
            save([expertWeights, topKIndices.indices, topKWeights.values])
            return {
                value: topKWeights.values,
                gradFunc: (dy, saved) => {
                    const [expertWeights, topKIndices, topKValues] = saved

                    // console.log('dy: ', dy.shape)
                    // console.log('expertWeights: ', expertWeights.shape)
                    // console.log('topKIndices: ', topKIndices.shape)
                    // console.log('topK values: ', topKValues.shape)

                    const tileShape = [1, 1, expertWeights.shape[2] / k]
                    return [
                        dy
                            .mul(topKValues.softmax())
                            .tile(tileShape)
                            .mul(expertWeights)
                    ]

                    // const inputReshaped = dy.reshape([
                    //     -1,
                    //     expertWeights.shape[2]
                    // ])
                    // console.log(inputReshaped)

                    // const projected = tf.matMul(
                    //     inputReshaped,
                    //     this.projectionMatrix.read(),
                    //     false,
                    //     true
                    // )
                    // console.log(projected)

                    // const projectionReshaped = projected.reshape(
                    //     expertWeights.shape
                    // )
                    // console.log(projectionReshaped)
                    // const mask = tf.zerosLike(expertWeights)
                    // const scatterIndices = tf
                    //     .range(0, dy.shape[0], 1, 'int32')
                    //     .reshape([-1, 1])
                    // const updateValues = tf.gather(topKValues, dy.shape[0] - 1)
                    // const updatedMask = tf.scatterND(
                    //     scatterIndices,
                    //     updateValues,
                    //     mask.shape
                    // )

                    // const weights = tf.zerosLike(expertWeights)
                    // const gradients = dy.tile([1, 1, this.numExperts / k])

                    // return [weights.mul(gradients.softmax())]
                    // console.log(this.projectionMatrix.read())
                    // const projectedDy = this.ops.applyDense(
                    //     this.projectionMatrix.read(),
                    //     dy
                    // )
                }
            }
            // return {
            //     value: topKWeights.values,
            //     gradFunc: (dy, saved) => {
            //         const [expertWeights] = saved
            //         // console.log(dy)
            //         // console.log(expertWeights)

            //         const tileShape = [1, 1, expertWeights.shape[2] / k]
            //         const scaleFactor = 1.0 / k
            //         return [
            //             dy
            //                 .tile(tileShape)
            //                 .mul(tf.scalar(scaleFactor))
            //                 .mul(expertWeights)
            //         ]
            //         // return [dy.tile(tileShape).mul(expertWeights.softmax())]
            //         // return [tf.clipByValue(expertWeights, -1e3, 1e3)]
            //         // return [expertWeights.tanh()]
            //         // return [expertWeights.softmax()] // most stable
            //     }
            // }
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
