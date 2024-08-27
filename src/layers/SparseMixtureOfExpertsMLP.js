import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SparseMixtureOfExpertsMLP extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts
        this.topK = config.topK || 2
        this.switchingDim = config.switchingDim || 128
        this.mlpDim = config.mlpDim || 256
        this.activation = config.activation || 'swish'
        this.temperature = config.temperature || 1.0
        this.epsilon = 1e-8
        this.expertUsage = tf.variable(tf.zeros([this.numExperts]), false)
        this.totalUsage = tf.variable(tf.scalar(0), false)
        this.debug = false
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.switchingHidden = this.addWeight(
            'switchingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ops.getSeed()
            })
        )
        this.switchingKernel = this.addWeight(
            'switchingKernel',
            [this.switchingDim, this.numExperts],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ops.getSeed()
            })
        )
        this.expertWeights = this.addWeight(
            'expertWeights',
            [this.numExperts, inputDim],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        this.inProjKernel = this.addWeight(
            `inProjKernel`,
            [this.numExperts, inputDim, this.mlpDim],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        this.outProjKernel = this.addWeight(
            `outProjKernel`,
            [this.numExperts, this.mlpDim, inputDim],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const switchingHidden = this.ops.applyDense(
                inputs,
                this.switchingHidden.read()
            )
            const switchingGate = tf.layers
                .activation({ activation: this.activation })
                .apply(switchingHidden)
            const switchingScores = this.ops.applyDense(
                switchingGate,
                this.switchingKernel.read()
            )

            const expertIndices = this.topKWithGumbel(
                switchingScores,
                this.topK
            )

            if (kwargs.training) this.computeUtilization(expertIndices)

            const rawIndices = expertIndices.arraySync()

            const [batchNum, timeSteps, features] = inputs.shape

            const batchOutputs = []
            for (let i = 0; i < batchNum; i++) {
                const stepOutputs = []
                for (let j = 0; j < timeSteps; j++) {
                    const topKOutputs = []
                    for (let k = 0; k < this.topK; k++) {
                        const expertIndex = rawIndices[i][j][k]
                        const inputSlice = inputs.slice([i, j, 0], [1, 1, -1])
                        const expertOutput = this.computeExpert(
                            inputSlice,
                            expertIndex
                        )
                        topKOutputs.push(expertOutput)
                    }
                    const averagedOutput = topKOutputs
                        .reduce((prev, curr, expertIndex) => {
                            return prev.add(curr)
                        }, tf.zeros(topKOutputs[0].shape))
                        .div(this.topK)
                    stepOutputs.push(averagedOutput)
                }
                batchOutputs.push(tf.concat(stepOutputs, 1))
            }
            return tf.concat(batchOutputs, 0)
        })
    }

    topKWithGumbel(scores, k) {
        const gumbel = this.ops.gumbelSoftmax(scores, this.temperature)
        const expertIndices = tf.customGrad((gumbel, save) => {
            const { indices, values } = tf.topk(gumbel, k)
            save([gumbel])
            return {
                value: indices,
                gradFunc: (dy, [gumbel]) => {
                    return [gumbel]
                }
            }
        })(gumbel)

        return expertIndices
    }

    computeExpert(inputs, idx) {
        const expertIn = this.inProjKernel
            .read()
            .slice([idx, 0, 0], [1, -1, -1])
            .squeeze()

        let outputs = this.ops.applyDense(inputs, expertIn)

        outputs = this.ops.rmsNorm(outputs)

        outputs = tf.layers
            .activation({ activation: this.activation })
            .apply(outputs)

        const expertOut = this.outProjKernel
            .read()
            .slice([idx, 0, 0], [1, -1, -1])
            .squeeze()
        outputs = this.ops.applyDense(outputs, expertOut)

        // Residual connection
        return tf.add(inputs, outputs)
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

tf.serialization.registerClass(SparseMixtureOfExpertsMLP)
