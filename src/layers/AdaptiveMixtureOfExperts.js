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
        this.epsilon = 1e-6
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
        this.expertWeights = this.addWeight(
            'expertWeights',
            [this.topK, inputDim],
            'float32',
            tf.initializers.ones()
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

            const switchingHidden = this.ops.applyDense(
                inputs,
                this.switchingHidden.read(),
                this.switchingHiddenBias.read()
            )
            const switchingNormalized = this.ops.rmsNorm(switchingHidden)
            const switchingActivated = tf.layers
                .activation({ activation: this.activation })
                .apply(switchingNormalized)
            const switchingScores = this.ops.applyDense(
                switchingActivated,
                this.switchingKernel.read(),
                this.switchingBias.read()
            )

            const { expertIndices, expertWeights } = this.topKWithGumbel(
                switchingScores,
                this.topK,
                kwargs
            )

            const batchOutputs = []
            for (let i = 0; i < inputs.shape[0]; i++) {
                const batchInputs = inputs.slice([i, 0], [1, -1])
                const expertOutputs = []
                for (let j = 0; j < this.topK; j++) {
                    const expertIndex = expertIndices[i][j]
                    // console.log(expertWeights)
                    const expertValue = expertWeights.slice(
                        [i, j, 0],
                        [1, 1, -1]
                    )
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
        })
    }

    topKWithGumbel(scores, k, kwargs) {
        let expertIndices
        const gumbel = this.ops.gumbelSoftmax(scores, this.temperature)
        const samples = tf.customGrad((logits, save) => {
            const { indices, values } = tf.topk(logits, k)
            const reducedIndices = tf.argMax(values, 1)
            const gatheredIndices = tf.gatherND(
                indices,
                reducedIndices.expandDims(1)
            )
            expertIndices = gatheredIndices.squeeze().arraySync()
            if (kwargs.training) this.computeUtilization(indices)
            save([logits, reducedIndices])
            return {
                value: values,
                gradFunc: (dy, saved) => {
                    const [logits, indices] = saved
                    // const gatheredGradients = tf.gatherND(dy, indices)
                    // console.log(gatheredGradients)
                    // const updatedGradients = tf.scatterND(
                    //     indices,
                    //     gatheredGradients,
                    //     logits.shape
                    // )
                    // console.log(dy)
                    // console.log(logits)
                    const tileShape = [1, 1, logits.shape[2] / k]
                    return [dy.tile(tileShape).add(logits)]
                }
            }
        })(gumbel)
        // console.log(samples)
        // console.log(this.expertWeights.read())
        const expertWeights = this.expertWeights
            .read()
            .mul(samples.mean(1).expandDims(-1))
            .softmax()
        // console.log(expertWeights)
        return { expertIndices, expertWeights }
    }

    updateExpertUsage(selectedExperts) {
        const batchUsage = tf.sum(
            tf.oneHot(selectedExperts, this.numExperts),
            0
        )
        this.expertUsage.assign(this.expertUsage.add(batchUsage))
        this.totalUsage.assign(this.totalUsage.add(tf.sum(batchUsage)))
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
